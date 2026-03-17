import numpy as np


def export_traj_race(file_paths: dict,
                     traj_race: np.ndarray) -> None:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function is used to export the generated trajectory into a file. The generated files get an unique UUID and a
    hash of the ggv diagram to be able to check it later.

    Inputs:
    file_paths:     paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    traj_race:      race trajectory [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    # export race trajectory
    header = "s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2"
    fmt = "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f"
    with open(file_paths["traj_race_export"], 'ab') as fh:
        np.savetxt(fh, traj_race, fmt=fmt, header=header, comments="#")


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
