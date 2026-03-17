import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import copy
import csv
import json
import time
import traceback

import cv2
import numpy as np
import yaml
import pkg_resources
import matplotlib.pyplot as plt
import configparser

import opt_mintime_traj
import trajectory_planning_helpers as tph
import helper_funcs_glob
from trajectory_generator_new.config_utils import load_map_defaults


def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def load_vehicle_params(module, opt_type, mintime_opts):
    parser = configparser.ConfigParser()
    pars = {}
    if not parser.read(os.path.join(module, "params", "racecar.ini")):
        raise ValueError("Specified config file does not exist or is empty!")

    pars["ggv_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
    pars["ax_max_machines_file"] = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
    pars["stepsize_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
    pars["reg_smooth_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
    pars["veh_params"] = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
    pars["vel_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))

    if opt_type == 'shortest_path':
        pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_shortest_path'))
    elif opt_type in ['mincurv', 'mincurv_iqp']:
        pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))
    elif opt_type == 'mintime':
        pars["curv_calc_opts"] = json.loads(parser.get('GENERAL_OPTIONS', 'curv_calc_opts'))
        pars["optim_opts"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
        pars["vehicle_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
        pars["tire_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'tire_params_mintime'))
        pars["pwr_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'pwr_params_mintime'))

        pars["optim_opts"]["var_friction"] = mintime_opts["var_friction"]
        pars["optim_opts"]["warm_start"] = mintime_opts["warm_start"]
        pars["vehicle_params_mintime"]["wheelbase"] = (
            pars["vehicle_params_mintime"]["wheelbase_front"]
            + pars["vehicle_params_mintime"]["wheelbase_rear"]
        )
    else:
        raise ValueError("Unknown optimization type!")

    return pars


def check_dependencies(module):
    requirements_path = os.path.join(module, 'requirements.txt')
    dependencies = []
    with open(requirements_path, 'r') as fh:
        for line in fh:
            dependencies.append(line.rstrip())
    try:
        pkg_resources.require(dependencies)
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as exc:
        print(f"WARNING: dependency check skipped: {exc}")


def map_to_pixel(x_m, y_m, height, res, origin):
    col = (x_m - origin[0]) / res
    row = height - (y_m - origin[1]) / res
    return int(round(col)), int(round(row))


def build_track_with_width(centerline_xy, dist_img, res, origin, fallback_width=0.6, width_scale=1.0):
    if dist_img is None:
        widths = np.full((centerline_xy.shape[0],), fallback_width * width_scale, dtype=float)
        return np.column_stack((centerline_xy, widths))
    h, w = dist_img.shape
    widths = []
    for x_m, y_m in centerline_xy:
        col, row = map_to_pixel(x_m, y_m, h, res, origin)
        col = min(max(col, 0), w - 1)
        row = min(max(row, 0), h - 1)
        d_px = dist_img[row, col]
        widths.append(max(2 * d_px * res, res) * width_scale)
    widths = np.array(widths)
    return np.column_stack((centerline_xy, widths))


def save_track_csv(track_xyz, path):
    with open(path, 'w') as f:
        f.write("#x_m,y_m,w_tr\n")
        for row in track_xyz:
            f.write(f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f}\n")


def resample_centerline(xy, min_points=200, smooth_window=7):
    if xy.shape[0] < 2:
        return xy
    # remove duplicate last point if closed
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    # cumulative arc length
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.insert(np.cumsum(d), 0, 0.0)
    total = s[-1]
    if total <= 0:
        return xy
    n = max(min_points, xy.shape[0])
    s_new = np.linspace(0.0, total, n, endpoint=False)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    xy_new = np.column_stack((x_new, y_new))
    # optional smoothing on closed curve
    if smooth_window and smooth_window > 2:
        k = np.ones(smooth_window) / smooth_window
        pad = smooth_window // 2
        xs = np.r_[xy_new[-pad:, 0], xy_new[:, 0], xy_new[:pad, 0]]
        ys = np.r_[xy_new[-pad:, 1], xy_new[:, 1], xy_new[:pad, 1]]
        xs = np.convolve(xs, k, mode="valid")
        ys = np.convolve(ys, k, mode="valid")
        xy_new = np.column_stack((xs, ys))
    # close loop
    xy_new = np.vstack((xy_new, xy_new[0]))
    return xy_new


def validate_raceline(raceline_xy, track_mask, res, origin):
    h, w = track_mask.shape
    violations = []
    for i, (x_m, y_m) in enumerate(raceline_xy):
        col, row = map_to_pixel(x_m, y_m, h, res, origin)
        if col < 0 or col >= w or row < 0 or row >= h:
            violations.append(i)
            continue
        if track_mask[row, col] == 0:
            violations.append(i)
    return violations


def run_optimization(track_file, pars, file_paths, imp_opts, opt_type, debug=False, plot_opts=None, plot_debug_mintime=False):
    start = time.time()
    reftrack_imp = helper_funcs_glob.src.import_track.import_track(
        imp_opts=imp_opts,
        file_path=track_file,
        width_veh=pars["veh_params"]["width"]
    )
    imp_opts_local = dict(imp_opts)
    total_width = reftrack_imp[:, 2] + reftrack_imp[:, 3]
    width_opt = pars["optim_opts"].get("width_opt", pars["veh_params"]["width"])
    min_width_raw = float(np.min(total_width))
    min_width_cap = max(min_width_raw - 0.05, pars["veh_params"]["width"] * 1.05)
    if width_opt > min_width_cap:
        pars["optim_opts"]["width_opt"] = min_width_cap
        width_opt = min_width_cap
        if debug:
            print(f"[opt] capped width_opt to {width_opt:.2f}m (min track {min_width_raw:.2f}m)")
    min_total_width = width_opt + 0.1
    deficit = min_total_width - total_width
    if np.any(deficit > 0):
        add = np.maximum(deficit, 0) / 2.0
        reftrack_imp[:, 2] = reftrack_imp[:, 2] + add
        reftrack_imp[:, 3] = reftrack_imp[:, 3] + add
        if debug:
            print(f"[opt] widened {np.count_nonzero(deficit > 0)} points to min width {min_total_width:.2f}m")
    if opt_type == "mintime":
        min_width_raw = float(np.min(reftrack_imp[:, 2] + reftrack_imp[:, 3]))
        width_opt = min(width_opt, max(min_width_raw - 0.05, pars["veh_params"]["width"] * 1.05))
        pars["optim_opts"]["width_opt"] = width_opt
        if imp_opts_local["min_track_width"] is None:
            imp_opts_local["min_track_width"] = width_opt + 0.05
        else:
            imp_opts_local["min_track_width"] = max(imp_opts_local["min_track_width"], width_opt + 0.05)

    try:
        reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp = \
            helper_funcs_glob.src.prep_track.prep_track(
                reftrack_imp=reftrack_imp,
                reg_smooth_opts=pars["reg_smooth_opts"],
                stepsize_opts=pars["stepsize_opts"],
                debug=debug,
                min_width=imp_opts_local["min_track_width"]
            )
    except Exception as exc:
        msg = str(exc)
        if ("spline normals are crossed" not in msg
                and "normals is crossed" not in msg
                and "pair of normals is crossed" not in msg):
            raise
        # fallback: skip normals crossing check
        reftrack_interp = tph.spline_approximation.spline_approximation(
            track=reftrack_imp,
            k_reg=pars["reg_smooth_opts"]["k_reg"],
            s_reg=pars["reg_smooth_opts"]["s_reg"],
            stepsize_prep=pars["stepsize_opts"]["stepsize_prep"],
            stepsize_reg=pars["stepsize_opts"]["stepsize_reg"],
            debug=debug
        )
        refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
        coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.calc_splines(
            path=refpath_interp_cl
        )

    if opt_type == 'mincurv':
        alpha_opt = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack_interp,
            normvectors=normvec_normalized_interp,
            A=a_interp,
            kappa_bound=pars["veh_params"]["curvlim"],
            w_veh=pars["optim_opts"]["width_opt"],
            print_debug=debug,
            plot_debug=plot_opts["mincurv_curv_lin"] if plot_opts else False
        )[0]
    elif opt_type == 'mincurv_iqp':
        # iqp handler signature differs across versions; keep mincurv only in this script
        raise RuntimeError("mincurv_iqp disabled in multi mode due to version mismatch")
    elif opt_type == 'mintime':
        # ensure safe trajectory limits are set if required
        if pars["optim_opts"].get("safe_traj"):
            if (pars["optim_opts"].get("ax_pos_safe") is None or
                    pars["optim_opts"].get("ax_neg_safe") is None or
                    pars["optim_opts"].get("ay_safe") is None):
                ggv, _ = tph.import_veh_dyn_info.import_veh_dyn_info(
                    ggv_import_path=file_paths["ggv_file"],
                    ax_max_machines_import_path=file_paths["ax_max_machines_file"]
                )
                pars["optim_opts"]["ax_pos_safe"] = np.amin(ggv[:, 1])
                pars["optim_opts"]["ax_neg_safe"] = -np.amin(ggv[:, 1])
                pars["optim_opts"]["ay_safe"] = np.amin(ggv[:, 2])
        alpha_opt, v_opt, reftrack_interp, a_interp_tmp, normvec_normalized_interp = \
            opt_mintime_traj.src.opt_mintime.opt_mintime(
                reftrack=reftrack_interp,
                coeffs_x=coeffs_x_interp,
                coeffs_y=coeffs_y_interp,
                normvectors=normvec_normalized_interp,
                pars=pars,
                tpamap_path=file_paths["tpamap"],
                tpadata_path=file_paths["tpadata"],
                export_path=file_paths["mintime_export"],
                print_debug=debug,
                plot_debug=plot_debug_mintime
            )
        if a_interp_tmp is not None:
            a_interp = a_interp_tmp
        v_opt = np.asarray(v_opt).reshape(-1)
    else:
        raise ValueError("Unknown opt_type")

    alpha_opt = np.asarray(alpha_opt).reshape(-1)
    if debug:
        print(f"[opt] {opt_type} solved in {time.time() - start:.2f}s")

    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, \
        s_points_opt_interp, spline_lengths_opt, el_lengths_opt_interp = tph.create_raceline.create_raceline(
            refline=reftrack_interp[:, :2],
            normvectors=normvec_normalized_interp,
            alpha=alpha_opt,
            stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"]
        )

    psi_vel_opt, kappa_opt = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt,
        coeffs_y=coeffs_y_opt,
        ind_spls=spline_inds_opt_interp,
        t_spls=t_vals_opt_interp
    )

    # Adjust psi by 90 degrees (pi/2 radians) ############################
    psi_vel_opt = psi_vel_opt + np.pi / 2 + (2*np.pi)
    psi_vel_opt= psi_vel_opt % (2*np.pi)


    # velocity profile
    ggv, ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
        ggv_import_path=file_paths["ggv_file"],
        ax_max_machines_import_path=file_paths["ax_max_machines_file"]
    )

    if opt_type == 'mintime':
        s_splines = np.cumsum(spline_lengths_opt)
        s_splines = np.insert(s_splines, 0, 0.0)
        vx_profile_opt = np.interp(s_points_opt_interp, s_splines[:-1], v_opt)
    else:
        vx_profile_opt = tph.calc_vel_profile.calc_vel_profile(
            ggv=ggv,
            ax_max_machines=ax_max_machines,
            v_max=pars["veh_params"]["v_max"],
            kappa=kappa_opt,
            el_lengths=el_lengths_opt_interp,
            closed=True,
            filt_window=pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=pars["vel_calc_opts"]["dyn_model_exp"],
            drag_coeff=pars["veh_params"]["dragcoeff"],
            m_veh=pars["veh_params"]["mass"]
        )

    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    ax_profile_opt = tph.calc_ax_profile.calc_ax_profile(
        vx_profile=vx_profile_opt_cl,
        el_lengths=el_lengths_opt_interp,
        eq_length_output=False
    )
    t_profile_cl = tph.calc_t_profile.calc_t_profile(
        vx_profile=vx_profile_opt,
        ax_profile=ax_profile_opt,
        el_lengths=el_lengths_opt_interp
    )
    laptime = t_profile_cl[-1]

    trajectory_opt = np.column_stack((
        s_points_opt_interp, raceline_interp, psi_vel_opt, kappa_opt, vx_profile_opt, ax_profile_opt
    ))

    traj_race_cl = np.vstack((trajectory_opt, trajectory_opt[0, :]))
    traj_race_cl[-1, 0] = np.sum(spline_lengths_opt)
    return laptime, traj_race_cl, raceline_interp


def run_all_centerlines(centerline_files, centerline_dir, args, module, file_paths, imp_opts, plot_opts,
                        dist_img, res, origin, track_img, track_mask, mintime_opts, clockwise=False):
    results = []
    racelines = []
    overlay_base = None
    if track_img is not None:
        overlay_base = cv2.cvtColor(track_img, cv2.COLOR_GRAY2BGR)
    out_dir = os.path.join(module, "outputs", args.map, "multi_mintime")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "fail_logs")
    if args.save_solver_log:
        os.makedirs(log_dir, exist_ok=True)

    total = len(centerline_files)
    t0 = time.time()
    for idx, fname in enumerate(centerline_files):
        print(f"[{idx+1}/{total}] processing {fname}...")
        in_path = os.path.join(centerline_dir, fname)
        data = np.loadtxt(in_path, comments="#", delimiter=",")
        if data.shape[1] >= 3:
            orig_xy = data[:, :2]
            if np.allclose(orig_xy[0], orig_xy[-1]):
                orig_xy = orig_xy[:-1]
                w_tr = data[:-1, 2]
            else:
                w_tr = data[:, 2]
            if clockwise:
                orig_xy = np.flipud(orig_xy)
                w_tr = np.flipud(w_tr)
            d = np.sqrt(np.sum(np.diff(orig_xy, axis=0) ** 2, axis=1))
            s = np.insert(np.cumsum(d), 0, 0.0)
            cl_xy = resample_centerline(orig_xy, min_points=200, smooth_window=args.smooth_window)
            d2 = np.sqrt(np.sum(np.diff(cl_xy, axis=0) ** 2, axis=1))
            s2 = np.insert(np.cumsum(d2), 0, 0.0)
            w_new = np.interp(s2, s, w_tr) * args.width_scale
            track_xyz = np.column_stack((cl_xy, w_new))
        else:
            cl_xy = data[:, :2]
            if clockwise:
                cl_xy = np.flipud(cl_xy)
            cl_xy = resample_centerline(cl_xy, min_points=200, smooth_window=args.smooth_window)
            track_xyz = build_track_with_width(cl_xy, dist_img, res, origin, width_scale=args.width_scale)
        if args.min_track_width is not None and track_xyz.shape[1] >= 3:
            track_xyz[:, 2] = np.maximum(track_xyz[:, 2], args.min_track_width)

        track_file = os.path.join(out_dir, f"track_{idx:02d}.csv")
        save_track_csv(track_xyz, track_file)

        base_mintime_opts = {
            "ipopt_max_iter": args.ipopt_max_iter,
            "ipopt_print_level": args.ipopt_print_level,
            "ipopt_print_time": args.ipopt_print_time,
        }
        if args.ipopt_tol is not None:
            base_mintime_opts["ipopt_tol"] = args.ipopt_tol
        attempts = [
            ("mintime", {"optim_opts": base_mintime_opts}),
            ("mintime", {"optim_opts": {"safe_traj": True, **base_mintime_opts}}),
            ("mintime", {"reg_smooth_opts": {"s_reg": 20}, "stepsize_opts": {"stepsize_prep": 0.5, "stepsize_reg": 0.3}, "optim_opts": base_mintime_opts}),
            ("mintime", {"reg_smooth_opts": {"s_reg": 80}, "stepsize_opts": {"stepsize_prep": 0.2, "stepsize_reg": 0.2}, "optim_opts": base_mintime_opts}),
            ("mincurv", {"optim_opts": {"width_opt": 0.3}}),
            ("mincurv", {"optim_opts": {"width_opt": 0.2}}),
            ("mincurv", {"reg_smooth_opts": {"s_reg": 80}, "stepsize_opts": {"stepsize_prep": 0.2, "stepsize_reg": 0.2}, "optim_opts": {"width_opt": 0.2}}),
        ]
        best = None
        attempt_logs = []
        for attempt_idx, (opt_type, overrides) in enumerate(attempts):
            try:
                print(f"  -> trying {opt_type} {overrides if overrides else ''}".rstrip())
                pars = load_vehicle_params(module, opt_type, mintime_opts)
                if "optim_opts" in overrides:
                    pars["optim_opts"].update(overrides["optim_opts"])
                if "reg_smooth_opts" in overrides:
                    pars["reg_smooth_opts"].update(overrides["reg_smooth_opts"])
                if "stepsize_opts" in overrides:
                    pars["stepsize_opts"].update(overrides["stepsize_opts"])
                log_file = None
                if args.save_solver_log and opt_type == "mintime":
                    log_file = os.path.join(
                        log_dir, f"centerline_{idx:02d}_attempt_{attempt_idx}_{opt_type}.log"
                    )
                    pars["optim_opts"]["ipopt_output_file"] = log_file
                    pars["optim_opts"]["ipopt_print_level"] = max(
                        pars["optim_opts"].get("ipopt_print_level", 0), 5
                    )
                    pars["optim_opts"]["ipopt_file_print_level"] = pars["optim_opts"]["ipopt_print_level"]
                    pars["optim_opts"]["ipopt_print_time"] = True

                laptime, traj_race_cl, raceline = run_optimization(
                    track_file,
                    pars,
                    file_paths,
                    imp_opts,
                    opt_type,
                    debug=False,
                    plot_opts=plot_opts,
                    plot_debug_mintime=args.plot_debug
                )
                if not np.isfinite(laptime):
                    attempt_logs.append({
                        "opt_type": opt_type,
                        "overrides": overrides,
                        "status": "invalid_laptime",
                        "log_file": log_file,
                    })
                    continue
                best = (opt_type, laptime, traj_race_cl, raceline)
                print(f"  -> success {opt_type} laptime={laptime:.3f}")
                break
            except Exception as exc:
                print(f"  -> failed {opt_type}: {exc}")
                attempt_logs.append({
                    "opt_type": opt_type,
                    "overrides": overrides,
                    "status": "exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                    "log_file": log_file,
                })
                continue

        if best is None:
            print(f"[{idx+1}/{total}] no valid solution for {fname}")
            results.append({"idx": idx, "status": "failed", "attempts": attempt_logs})
            continue

        opt_type, laptime, traj_race_cl, raceline = best
        violations = []
        if track_img is not None:
            violations = validate_raceline(raceline, track_mask, res, origin)
        results.append({
            "idx": idx,
            "status": "ok",
            "opt_type": opt_type,
            "laptime": float(laptime),
            "violations": len(violations),
            "attempts": attempt_logs,
        })
        racelines.append((idx, raceline, laptime, violations))
        traj_path = os.path.join(out_dir, f"traj_race_cl_{idx:02d}.csv")
        helper_funcs_glob.src.export_traj_race.export_traj_race(
            file_paths={"traj_race_export": traj_path}, traj_race=traj_race_cl
        )
        print(f"[{idx+1}/{total}] done {fname} (opt={opt_type}, laptime={laptime:.3f})")

        if args.plot_mincurv and opt_type == "mincurv" and overlay_base is not None:
            overlay = overlay_base.copy()
            pts = []
            for x_m, y_m in raceline:
                col, row = map_to_pixel(x_m, y_m, overlay.shape[0], res, origin)
                pts.append((col, row))
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], (255, 0, 0), 1)
            mincurv_path = os.path.join(out_dir, f"mincurv_overlay_{idx:02d}.png")
            cv2.imwrite(mincurv_path, overlay)
            if not args.headless:
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                plt.title(f"MinCurv overlay idx={idx:02d}")
                plt.axis("off")
                plt.show()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Summary results:")
    print(json.dumps(results, indent=2))

    ranking = [r for r in results if r.get("status") == "ok" and np.isfinite(r.get("laptime", np.inf))]
    ranking.sort(key=lambda r: r["laptime"])
    rank_path = os.path.join(out_dir, "mintime_ranking.csv")
    with open(rank_path, "w") as f:
        f.write("rank,centerline_idx,laptime_s,opt_type\n")
        for i, r in enumerate(ranking, start=1):
            f.write(f"{i},{r['idx']},{r['laptime']:.6f},{r['opt_type']}\n")
    print(f"Saved mintime ranking: {rank_path}")
    print(f"Total time: {time.time() - t0:.2f}s")

    if racelines:
        best_idx, best_raceline, best_time, _ = min(racelines, key=lambda x: x[2])
        plt.figure(figsize=(10, 6))
        for idx, line, _, _ in racelines:
            plt.plot(line[:, 0], line[:, 1], linewidth=1.0)
        plt.plot(best_raceline[:, 0], best_raceline[:, 1], linewidth=2.5)
        plt.axis('equal')
        plt.title(f"Best mintime idx={best_idx} t={best_time:.2f}s")
        plt.savefig(os.path.join(out_dir, "racelines_all.png"), dpi=200)
        if not args.headless:
            plt.show()

    if overlay_base is not None and racelines:
        overlay = overlay_base.copy()
        colors = [
            (0, 0, 255), (0, 255, 255), (0, 255, 0),
            (255, 255, 0), (255, 0, 0), (255, 0, 255)
        ]
        for idx, line, _, violations in racelines:
            color = colors[idx % len(colors)]
            pts = []
            for x_m, y_m in line:
                col, row = map_to_pixel(x_m, y_m, overlay.shape[0], res, origin)
                pts.append((col, row))
            for i in range(len(pts) - 1):
                cv2.line(overlay, pts[i], pts[i + 1], color, 1)
            for vidx in violations:
                if 0 <= vidx < len(pts):
                    cv2.circle(overlay, pts[vidx], 2, (0, 0, 0), -1)
        cv2.imwrite(os.path.join(out_dir, "racelines_overlay.png"), overlay)

    any_mintime = any(r.get("opt_type") == "mintime" for r in results if r.get("status") == "ok")
    return any_mintime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-centerline global trajectory optimization (mintime best)")
    parser.add_argument("--map", type=str, default=None, help="Map name (without extension)")
    parser.add_argument("--centerline-map", type=str, default=None, help="Map name used for centerlines")
    parser.add_argument("--headless", action="store_true", help="Disable plotting windows")
    parser.add_argument("--plot-debug", action="store_true", help="Enable mintime debug plots (needs display)")
    parser.add_argument("--plot-mincurv", action="store_true", help="Show/save mincurv debug plots on fallback")
    parser.add_argument("--save-solver-log", action="store_true", help="Save IPOPT solver logs for mintime attempts")
    parser.add_argument("--ipopt-max-iter", type=int, default=1200, help="IPOPT max iterations for mintime")
    parser.add_argument("--ipopt-print-level", type=int, default=0, help="IPOPT print level for mintime")
    parser.add_argument("--ipopt-print-time", action="store_true", help="Enable IPOPT timing output")
    parser.add_argument("--ipopt-tol", type=float, default=None, help="IPOPT tolerance for mintime (optional)")
    parser.add_argument("--width-scale", type=float, default=0.6, help="Scale factor for track width")
    parser.add_argument("--smooth-window", type=int, default=7, help="Centerline smoothing window (odd)")
    parser.add_argument("--min-track-width", type=float, default=None, help="Force minimum total track width (m)")
    parser.add_argument("--sweep", action="store_true", help="Sweep parameters until mintime succeeds")
    args = parser.parse_args()
    if args.headless:
        plt.switch_backend("Agg")

    module = ROOT
    config_file = os.path.join(module, "config", "params.yaml")
    cfg = read_yaml(config_file)
    clockwise = bool(cfg.get("clockwise", False))
    if clockwise:
        print("INFO: clockwise=True -> reversing centerline order for mintime optimization")

    default_map, _ = load_map_defaults(module)
    map_name = args.map or default_map
    centerline_map = args.centerline_map or f"{map_name}_processed"

    # use processed map for widths
    map_yaml = read_yaml(os.path.join(module, "maps", f"{centerline_map}.yaml"))
    res = map_yaml["resolution"]
    origin = map_yaml["origin"]

    map_img_path = os.path.join(module, "maps", f"{centerline_map}.pgm")
    track_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
    if track_img is None:
        alt_path = os.path.join(module, "maps", f"{centerline_map}.png")
        track_img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
        if track_img is not None:
            map_img_path = alt_path
    dist_img = None
    track_mask = None
    if track_img is None:
        print(f"WARNING: Map image not found: {map_img_path} (using fallback width)")
    else:
        track_mask = (track_img > 127).astype(np.uint8) * 255
        dist_img = cv2.distanceTransform(track_mask, cv2.DIST_L2, 5)

    # load centerlines
    centerline_dir = os.path.join(module, "outputs", centerline_map, "multi_centerlines")
    centerline_files = sorted(
        f for f in os.listdir(centerline_dir) if f.startswith("centerline_") and f.endswith(".csv")
    )
    if not centerline_files:
        raise FileNotFoundError(f"No centerline_*.csv in {centerline_dir}")

    # dependencies
    check_dependencies(module)

    # optimization options
    mintime_opts = {
        "tpadata": None,
        "warm_start": False,
        "var_friction": None,
        "reopt_mintime_solution": False,
        "recalc_vel_profile_by_tph": False
    }
    imp_opts = {
        "flip_imp_track": False,
        "set_new_start": False,
        "new_start": np.array([0.0, -47.0]),
        "min_track_width": args.min_track_width,
        "num_laps": 1
    }
    plot_opts = {"mincurv_curv_lin": args.plot_mincurv}

    # file paths
    file_paths = {
        "module": module,
    }
    file_paths["ggv_file"] = os.path.join(module, "inputs", "veh_dyn_info", "ggv.csv")
    file_paths["ax_max_machines_file"] = os.path.join(module, "inputs", "veh_dyn_info", "ax_max_machines.csv")
    file_paths["tpamap"] = os.path.join(module, "inputs", "frictionmaps", f"{map_name}_tpamap.csv")
    file_paths["tpadata"] = os.path.join(module, "inputs", "frictionmaps", f"{map_name}_tpadata.json")
    file_paths["mintime_export"] = os.path.join(module, "outputs", map_name, "mintime")

    os.makedirs(file_paths["mintime_export"], exist_ok=True)

    if args.sweep:
        sweep_width_scales = [0.8, 1.0, 1.2]
        sweep_smooth = [7, 9, 11]
        for ws in sweep_width_scales:
            for sw in sweep_smooth:
                args.width_scale = ws
                args.smooth_window = sw
                print(f"=== sweep width_scale={ws} smooth_window={sw} ===")
                if run_all_centerlines(centerline_files, centerline_dir, args, module, file_paths, imp_opts,
                                       plot_opts, dist_img, res, origin, track_img, track_mask, mintime_opts,
                                       clockwise=clockwise):
                    print("Mintime succeeded; stopping sweep.")
                    raise SystemExit(0)
        print("Sweep finished without mintime success.")
    else:
        run_all_centerlines(centerline_files, centerline_dir, args, module, file_paths, imp_opts,
                            plot_opts, dist_img, res, origin, track_img, track_mask, mintime_opts,
                            clockwise=clockwise)
