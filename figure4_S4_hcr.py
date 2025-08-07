import h5py
import numpy as np
import pandas as pd
from multifeature_integration_paper.figure_helper import Figure
from scipy.stats import ttest_ind
import nrrd
from matplotlib.colors import ListedColormap

def create_locs_subplots(fig, x_l=10.5, y_t=8.5, x_bs=5.4, y_bs = 3.5, wta=False):

    motion_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l-2.75, legend_ypos=y_t+2.75)
    lumi_locs_plot = fig.create_plot(xpos=x_l+x_bs, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=x_l+x_bs-2.75, legend_ypos=y_t+2.75)

    if not wta:
        dark_locs_plot = fig.create_plot(xpos=x_l, ypos=y_t - y_bs, plot_height=2, plot_width=2, axis_off=True,
                                         xmin=30, xmax=800, ymin=850, ymax=80,
                                         legend_xpos=x_l - 2.75, legend_ypos=y_t - y_bs + 2.75)
        bright_locs_plot = fig.create_plot(xpos=x_l + x_bs, ypos=y_t - y_bs, plot_height=2, plot_width=2, axis_off=True,
                                           xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l + x_bs - 2.75, legend_ypos=y_t - y_bs + 2.75)
        drive_locs_plot = fig.create_plot(xpos=x_l+x_bs+x_bs, ypos=y_t-y_bs, plot_height=2, plot_width=2, axis_off=True,
                                              xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l+x_bs-2.75, legend_ypos=y_t-y_bs-y_bs+2.75)
        diff_locs_plot = fig.create_plot(xpos=x_l+x_bs+x_bs, ypos=y_t, plot_height=2, plot_width=2, axis_off=True,
                                              xmin=30, xmax=800, ymin=850, ymax=80,
                                           legend_xpos=x_l-2.75, legend_ypos=y_t-y_bs-y_bs+2.75)

        subfigs_locs = [motion_locs_plot, lumi_locs_plot, dark_locs_plot, bright_locs_plot, drive_locs_plot, diff_locs_plot]
    else:
        subfigs_locs = [motion_locs_plot, lumi_locs_plot,]


    return subfigs_locs

def create_traces_simple_subplots(fig, x_l=7.5, y_t=10., x_ss=1, y_ss=1, ymax_extra=0):

    traces_plota = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_ss-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.45, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    traces_plots = [traces_plota, traces_plotb, traces_plotc, traces_plotd,
                    traces_plote, traces_plotf, traces_plotg, traces_ploth,
                    traces_ploti]
    return traces_plots

def create_traces_subplots(fig, x_l=7.5, y_t=10., x_ss=1, x_bs=5.40, y_ss=1, y_bs=3.5, wta=False, ymax_extra=0):

    motion_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=1 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    motion_traces_plots = [motion_traces_plota, motion_traces_plotb, motion_traces_plotc, motion_traces_plotd,
                           motion_traces_plote, motion_traces_plotf, motion_traces_plotg, motion_traces_ploth,
                           motion_traces_ploti]

    lumi_traces_plota = fig.create_plot(xpos=x_l+x_bs, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotb = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotc = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotd = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plote = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotf = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plotg = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_ploth = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_ploti = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                          xmin=-5, xmax=62, ymin=-0.35, ymax=0.8 + ymax_extra,
                                          vspans=[[10, 40, 'lightgray', 1.0], ])
    lumi_traces_plots = [lumi_traces_plota, lumi_traces_plotb, lumi_traces_plotc, lumi_traces_plotd,
                        lumi_traces_plote, lumi_traces_plotf, lumi_traces_plotg, lumi_traces_ploth,
                        lumi_traces_ploti]

    if not wta:
        dark_traces_plota = fig.create_plot(xpos=x_l, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotb = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotc = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotd = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plote = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotf = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plotg = fig.create_plot(xpos=x_l, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_ploth = fig.create_plot(xpos=x_l+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_ploti = fig.create_plot(xpos=x_l+x_ss+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        dark_traces_plots = [dark_traces_plota, dark_traces_plotb, dark_traces_plotc, dark_traces_plotd,
                               dark_traces_plote, dark_traces_plotf, dark_traces_plotg, dark_traces_ploth,
                               dark_traces_ploti]

        bright_traces_plota = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotb = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotc = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotd = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plote = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotf = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plotg = fig.create_plot(xpos=x_l+x_bs, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_ploth = fig.create_plot(xpos=x_l+x_bs+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_ploti = fig.create_plot(xpos=x_l+x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        bright_traces_plots = [bright_traces_plota, bright_traces_plotb, bright_traces_plotc, bright_traces_plotd,
                             bright_traces_plote, bright_traces_plotf, bright_traces_plotg, bright_traces_ploth,
                             bright_traces_ploti]

        drive_traces_plota = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotb = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotc = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t-y_bs, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotd = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plote = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotf = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plotg = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_ploth = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_ploti = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t-y_bs-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                              xmin=-5, xmax=62, ymin=-0.35, ymax=1.1 + ymax_extra,
                                              vspans=[[10, 40, 'lightgray', 1.0], ])
        drive_traces_plots = [drive_traces_plota, drive_traces_plotb, drive_traces_plotc, drive_traces_plotd,
                               drive_traces_plote, drive_traces_plotf, drive_traces_plotg, drive_traces_ploth,
                               drive_traces_ploti]

        diff_traces_plota = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotb = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotc = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotd = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plote = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotf = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plotg = fig.create_plot(xpos=x_l+2*x_bs, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_ploth = fig.create_plot(xpos=x_l+2*x_bs+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_ploti = fig.create_plot(xpos=x_l+2*x_bs+x_ss+x_ss, ypos=y_t-y_ss-y_ss, plot_height=0.75, plot_width=0.75, axis_off=True,
                                            xmin=-5, xmax=62, ymin=-0.35, ymax=1,
                                            vspans=[[10, 40, 'lightgray', 1.0], ])
        diff_traces_plots = [diff_traces_plota, diff_traces_plotb, diff_traces_plotc, diff_traces_plotd,
                             diff_traces_plote, diff_traces_plotf, diff_traces_plotg, diff_traces_ploth,
                             diff_traces_ploti]

        subfigs_traces = [motion_traces_plots, lumi_traces_plots, dark_traces_plots, bright_traces_plots, drive_traces_plots, diff_traces_plots]
    else:
        subfigs_traces = [motion_traces_plots, lumi_traces_plots]

    return subfigs_traces

def subfig_HCR_example_stacks(example_GC_path_HCR, example_gad_path_HCR, example_Glut_path_HCR,
                              example_stack_GC_plot_HCR, example_stack_gad_vglut_plot_HCR,
                              example_loc_plot_overlap, example_loc_plot_GC, example_loc_plot_gad, example_loc_plot_Glut,
                              example_plane):
    GC_stack = nrrd.read(example_GC_path_HCR)[0]
    gad_stack = nrrd.read(example_gad_path_HCR)[0]
    Glut_stack = nrrd.read(example_Glut_path_HCR)[0]
    GC_low = 100
    GC_high = 6000
    gad_low = 50
    gad_high = 2000
    Glut_low = 250
    Glut_high = 500

    for z, offset in zip(range(5), [0, 50, 250, 300, 350]):
        example_stack_GC_plot_HCR.draw_image(np.clip(GC_stack[:, :, z*6].T, GC_low, GC_high), colormap='gray',
                           extent=(offset, 800 + offset, 800 + offset, offset), image_origin='upper')
        example_stack_GC_plot_HCR.draw_line([offset, 800 + offset, 800 + offset, offset, offset],
                          [offset, offset, 800 + offset, 800 + offset, offset], lc='w', lw=0.5)
    example_stack_GC_plot_HCR.draw_scatter([100, 150, 200, 900, 950, 1000], [900, 950, 1000, 100, 150, 200], ec='k', pc='k', ps=1)

    cmap_m = ListedColormap(np.c_[np.linspace(0, 1, 256), np.zeros(256), np.linspace(0, 1, 256)])
    cmap_g = ListedColormap(np.c_[np.zeros(256), np.linspace(0, 1, 256), np.zeros(256)])
    cmap_k = ListedColormap(np.c_[np.linspace(0, 1, 256), np.linspace(0, 1, 256), np.linspace(0, 1, 256)])

    for z, offset in zip(range(5), [0, 50, 250, 300, 350]):
        im = np.clip(gad_stack[:, :, z].T, gad_low, gad_high)
        im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
        im_m = cmap_m(im)

        im = np.clip(Glut_stack[:, :, z].T, Glut_low, Glut_high)
        im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
        im_g = cmap_g(im)

        im_combined = np.sum(np.stack([im_m, im_g], axis=3), axis=3)
        im_combined[im_combined > 1] = 1

        example_stack_gad_vglut_plot_HCR.draw_image(im_combined, extent=(offset, 800 + offset, 800 + offset, offset),
                                                    image_origin='upper')
        example_stack_gad_vglut_plot_HCR.draw_line([offset, 800 + offset, 800 + offset, offset, offset],
                          [offset, offset, 800 + offset, 800 + offset, offset], lc='w', lw=0.5)
    example_stack_gad_vglut_plot_HCR.draw_scatter([100, 150, 200, 900, 950, 1000], [900, 950, 1000, 100, 150, 200], ec='k', pc='k', ps=1)


    im = np.clip(GC_stack[:, :, example_plane].T, GC_low, GC_high)
    im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
    im_k = cmap_k(im)

    im = np.clip(gad_stack[:, :, example_plane].T, gad_low, gad_high)
    im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
    im_m = cmap_m(im)

    im = np.clip(Glut_stack[:, :, example_plane].T, Glut_low, Glut_high)
    im = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
    im_g = cmap_g(im)

    im_combined = np.sum(np.stack([im_k, im_m, im_g], axis=3), axis=3)
    im_combined[im_combined > 1] = 1

    example_loc_plot_overlap.draw_image(im_combined, extent=(0, 800, 800, 0), image_origin='upper')
    example_loc_plot_overlap.draw_line([780-168, 780], [780, 780], lc='w')
    example_loc_plot_overlap.draw_text(780-84, 730, '50\u00b5m', textcolor='w')

    for avg_im, subfig, name, color, alpha, clip_low, clip_high in zip([GC_stack[:, :, example_plane], gad_stack[:, :, example_plane], Glut_stack[:, :, example_plane]],
                                                  [example_loc_plot_GC, example_loc_plot_gad, example_loc_plot_Glut],
                                                  ['G8s', 'gad', 'vglut'],
                                                  ['k', 'm', 'g',],
                                                  [1.0, 0.9, 0.8],
                                                  [GC_low, gad_low, Glut_low],
                                                  [GC_high, gad_high, Glut_high]
                                                  ):
        subfig.draw_image(np.clip(avg_im.T, clip_low, clip_high), colormap='gray', extent=(0, 800, 800, 0), image_origin='upper')
        subfig.draw_text(400, -80, name, textcolor=color)
    return


def sub_plot_example_neuron(example_data_path, file_base_name, subfiga, subfigb, subfigc, subfigds, cell_name, cell_x_v, cell_y_v, z_plane_v, cell_x, cell_y, z_plane, tile):
    gad, _ = nrrd.read(fr'{example_data_path}\gadregistered2volume.nrrd', index_order='C')
    vglut, _ = nrrd.read(fr'{example_data_path}\vglutregistered2volume.nrrd', index_order='C')
    ref2, _ = nrrd.read(fr'{example_data_path}\vglutGcampregistered2volume.nrrd', index_order='C')
    _, volume_header = nrrd.read(fr'{example_data_path}\volume.nrrd',index_order='C')
    ref1, ref1_header = nrrd.read(fr'{example_data_path}\ref1.nrrd',index_order='C')

    #np.array(header["spacings"])
    scalev = np.linalg.norm(volume_header["space directions"], axis=0)[0]  # Pixel size in volume
    scalev1 = np.linalg.norm(ref1_header["space directions"], axis=0)[0]  # Pixel size in ref1

    # Define the window size for the cropped slice (e.g., 40x40 around the point)
    crop_size = 100

    print(cell_x_v / scalev, cell_y_v / scalev, cell_x, cell_y, tile)
    # For gad, vglut, ref2, use cell_x_v, cell_y_v, z_plane_v
    x_min_v = int(max(cell_x_v / scalev - crop_size // (2), 0))
    x_max_v = int(min(cell_x_v / scalev + crop_size // (2), gad.shape[2]))  # Assuming all volumes have same shape
    y_min_v = int(max(cell_y_v / scalev - crop_size // (2), 0))
    y_max_v = int(min(cell_y_v / scalev + crop_size // (2), gad.shape[1]))  # Assuming all volumes have same shape

    # For ref1, use cell_x, cell_y, z_plane
    x_min = int(max(cell_x - crop_size * scalev // (2 * scalev1), 0))
    x_max = int(min(cell_x + crop_size * scalev // (2 * scalev1), ref1.shape[2]))  # Assuming all volumes have same shape
    y_min = int(max(cell_y - crop_size * scalev // (2 * scalev1), 0))
    y_max = int(min(cell_y + crop_size * scalev // (2 * scalev1), ref1.shape[1]))  # Assuming all volumes have same shape

    z = str(z_plane).zfill(3)
    tile = str(tile).zfill(3)

    # Load contour from HDF5
    with h5py.File(
            fr'{example_data_path}\{file_base_name}\{file_base_name}_preprocessed_data.h5',
            'r',
    ) as f:
        print(f'repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{cell_name}')
        contour = f[f'repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours/{cell_name}'][:]
        contourv = np.floor(f[f'repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_contours_ants_volume_registered/{cell_name}'][:] / scalev).astype(int)
        traces = [[]] * 9
        for t, stim_name in enumerate(['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):
            trace = np.array(f[f'repeat00_tile{tile}_z{z}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics'][stim_name]['F'])[:, int(cell_name) - 10000, :]
            f0 = np.nanmean(trace, axis=1)
            dff0 = (trace - f0[:, None]) / f0[:, None]
            traces[t] = dff0

    # Adjust contour for cropping
    adjusted_contour = contour - np.array([x_min, y_min])
    adjusted_contourv = contourv[:, :2] - np.array([x_min_v, y_min_v])

    # Get slices from all four volumes for the given z_plane and the cropped region
    gad_slice = gad[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]
    vglut_slice = vglut[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]
    ref1_slice = ref1[int(z_plane), y_min:y_max, x_min:x_max]
    refvglut_slice = ref2[int(z_plane_v / 2), y_min_v:y_max_v, x_min_v:x_max_v]

    # Plot each slice in its own subplot
    # Plot gad base image
    # gad in magenta
    # Normalize images to [0, 1] if needed
    # gad normalization (to [0, 1])
    gad_min, gad_max = np.nanmin(gad_slice), np.nanmax(gad_slice)
    gad_norm = (gad_slice - gad_min) / (gad_max - gad_min + 1e-8)

    # vglut normalization (to [0, 1])
    vglut_min, vglut_max = np.nanmin(vglut_slice), np.nanmax(vglut_slice)
    vglut_norm = (vglut_slice - vglut_min) / (vglut_max - vglut_min + 1e-8)

    # Create RGB channels: magenta = R + B, green = G
    rgb = np.zeros((*gad_slice.shape, 3))
    rgb[..., 0] = gad_norm  # Red from gad
    rgb[..., 1] = vglut_norm  # Green from vglut
    rgb[..., 2] = gad_norm  # Blue from gad → magenta = red + blue

    # Display
    subfiga.draw_image(ref1_slice, colormap='gray', image_origin='lower', extent=(0, 100, 100, 0))
    subfiga.draw_line(adjusted_contour[:, 0], adjusted_contour[:, 1], lc='#D55E00', lw=0.5)
    subfiga.draw_line([20, 20+10/scalev], [90, 90], lc='w')
    subfiga.draw_text(20+5/scalev, 70, '10\u00b5m', textcolor='w')
    subfiga.draw_text(50, -10, f"G8s pre")

    subfigb.draw_image(refvglut_slice, colormap='gray', image_origin='lower', extent=(0, 100, 100, 0))
    subfigb.draw_line(adjusted_contourv[:, 0], adjusted_contourv[:, 1], lc='#D55E00', lw=0.5)
    subfigb.draw_text(50, -10, f"G8s post")

    subfigc.draw_image(rgb, image_origin='lower', extent=(0, 100, 100, 0))
    subfigc.draw_scatter(1 + crop_size // 2, 1 + crop_size // 2, pt='o', pc='#D55E00', ps=1, ec=None)
    subfigc.draw_text(50, -10, f"in situ")

    for subfig, trace in zip(subfigds, traces):
        subfig.draw_line(np.arange(0, 60, 0.5), np.nanmedian(trace, axis=0), yerr_pos=np.nanpercentile(trace, 75, axis=0) - np.nanmedian(trace, axis=0),
                         yerr_neg=np.nanmedian(trace, axis=0)-np.nanpercentile(trace, 25, axis=0),
                         lc='#D55E00', lw=1, eafc='#EEAE7C', eaalpha=1.0, ealw=1, eaec='#EEAE7C')

    subfigds[6].draw_line([-4, -4], [0, 0.5], lc='k')
    subfigds[6].draw_text(-20, 0., '0.5 dF/F\u2080', textlabel_rotation=90, textlabel_va="bottom")

    subfigds[8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigds[8].draw_text(50, -0.7, '20s')
    return

def get_avg_traces_per_neuron(file_paths, cell_names, z_planes, flip_stims):
    first_cell = True
    traces = [[]] * 9
    for file_path, cell_name, z_plane, flip_stim in zip(file_paths, cell_names, z_planes, flip_stims):
        with h5py.File(
                fr'{file_path}',
                'r',
        ) as f:

            if flip_stim:
                stim_names = ['lumi_right_dots_right', 'lumi_right_dots_left', 'lumi_right_dots_off', 'lumi_left_dots_right',
                              'lumi_left_dots_left', 'lumi_left_dots_off', 'lumi_off_dots_right', 'lumi_off_dots_left',
                              'lumi_off_dots_off']
            else:
                stim_names = ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off', 'lumi_right_dots_left',
                             'lumi_right_dots_right', 'lumi_right_dots_off', 'lumi_off_dots_left', 'lumi_off_dots_right',
                             'lumi_off_dots_off']
            for t, stim_name in enumerate(stim_names):
                trace = np.array(f[f'repeat00_tile000_z{z_plane:03d}_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics'][stim_name]['F'])[:, int(cell_name) - 10000, :]
                f0 = np.nanmean(trace, axis=1)
                dff0 = (trace - f0[:, None]) / f0[:, None]
                if first_cell:
                    traces[t] = np.nanmean(dff0, axis=0)
                else:
                    traces[t] = np.vstack([traces[t], np.nanmean(dff0, axis=0)])
                if first_cell and t==8:
                    first_cell = False

    return traces

def sub_plot_HCR_ratios(data_paths, file_names, csv_path, subfig_ratio, subfig_loc_gad, subfig_loc_glut,
                        subfigs_traces_gad, subfigs_traces_Glut, subfigs_locs_gad, subfigs_locs_Glut):
    all_cells = pd.DataFrame()

    for output_path, file in zip(data_paths, file_names):
        # File to save manual entries
        output_file_path = fr'{output_path}\manual_entries.csv'
        existing_entries = pd.read_csv(output_file_path)

        noi_df = pd.read_csv(fr'{output_path}\noi_df.csv')

        noi_df["cell_z_index"] = noi_df["cell_name"].astype(str) + "_" + noi_df["z_plane"].astype(str)
        existing_entries["cell_z_index"] = existing_entries["cell_name"].astype(str) + "_" + existing_entries[
            "zplane"].astype(str)

        merged_by_cell = pd.merge(noi_df, existing_entries, on="cell_z_index", how="inner")
        merged_by_cell["file_path"] = f'{output_path}\\{file}\\{file}_preprocessed_data.h5'

        merged_by_cell.loc[:, "description"] = merged_by_cell["description"].replace({
            "gaba weak": "gaba",
            "vglut weak": "vglut"
        })

        all_cells = pd.concat([all_cells, merged_by_cell])

    m_df = all_cells[(all_cells['mr'] == 1) | (all_cells['ml'] == 1)]
    dr_df = all_cells[(all_cells['drl'] == 1) | (all_cells['drr'] == 1)]
    l_df = all_cells[(all_cells['ll'] == 1) | (all_cells['lr'] == 1)]
    b_df = all_cells[(all_cells['bl'] == 1) | (all_cells['br'] == 1)]
    dk_df = all_cells[(all_cells['dkl'] == 1) | (all_cells['dkr'] == 1)]
    df_df = all_cells[(all_cells['dfl'] == 1) | (all_cells['dfr'] == 1)]

    # Define the dataframes corresponding to each region
    type_dfs = {
        "motion": m_df,
        "drive": dr_df,
        "luminance": l_df,
        "diff": df_df,
        "brightness": b_df,
        "dark": dk_df,
    }

    # Store results
    ratio_data = []

    for func_type, df in type_dfs.items():
        # Only keep gad or vglut-labeled cells
        filtered = df[df["description"].isin(["gaba", "vglut"])]
        if filtered.empty:
            continue

        # Group by file (fish) and description
        counts = filtered.groupby("file_path")["description"].value_counts().unstack(fill_value=0)

        # Compute ratio
        counts["vglut/gaba"] = counts.get("vglut", 0) / (counts.get("vglut", 0) + counts.get("gaba", 0) + 1e-6)

        # Store results
        for fish_file, row in counts.iterrows():
            ratio_data.append({
                "functional_type": func_type,
                "fish": fish_file,
                "ratio": row["vglut/gaba"]
            })

    # Convert to DataFrame
    ratio_df = pd.DataFrame(ratio_data)

    subfig_ratio.draw_line([-1, 6], [0.5, 0.5], lc='gray')

    # Jitter x-values slightly for better visibility
    for i, (func_type, color) in enumerate(zip(ratio_df["functional_type"].unique(),
                                             ['#359B73', '#2271B2', '#E69F00', '#D55E00', '#F748A5', '#9F0162',])):
        subset = ratio_df[ratio_df["functional_type"] == func_type]
        x_vals = [i + (np.random.rand() - 0.5) * 0.2 for _ in range(len(subset))]  # jitter
        subfig_ratio.draw_scatter(x_vals, subset["ratio"], ps=3, pc=np.repeat(color, 5), ec=None)  # size 70, black
        _, pval = ttest_ind(subset['ratio'], 0.5)
        if pval < 0.001:
            subfig_ratio.draw_text(i-0.5, 1, '***')
        elif pval < 0.01:
            subfig_ratio.draw_text(i-0.5, 1, '**')
        elif pval < 0.05:
            subfig_ratio.draw_text(i-0.5, 1, '*')
        print(func_type)
        print(ttest_ind(subset['ratio'], 0.5))

    data = pd.read_csv(csv_path)
    data['file_path'] = [data['file_path'][i].replace('Y:', 'W:') for i in data.index]
    gad_data = data[data['description'] == 'gaba']
    glut_data = data[data['description'] == 'vglut']

    for type, color, direction in zip(
            ['br', 'dkr', 'lr', 'mr', 'drr', 'dfr', 'bl', 'dkl', 'll', 'ml', 'drl', 'dfl', ],
            ['#F748A5', '#9F0162', '#E69F00', '#359B73', '#2271B2', '#D55E00', '#F748A5', '#9F0162',  '#E69F00', '#359B73', '#2271B2', '#D55E00'],
            ['r', 'r', 'r', 'r', 'r', 'r', 'l', 'l', 'l', 'l', 'l', 'l', ]):
        gad_type_data = gad_data[gad_data[type] == 1]
        glut_type_data = glut_data[glut_data[type] == 1]
        if direction == 'l':
            subfig_loc_gad.draw_scatter(gad_type_data['x_zbrain'].astype(float), gad_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            subfig_loc_gad.draw_scatter(gad_type_data['z_zbrain'].astype(float) + 515, gad_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            subfig_loc_glut.draw_scatter(glut_type_data['x_zbrain'].astype(float), glut_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            subfig_loc_glut.draw_scatter(glut_type_data['z_zbrain'].astype(float) + 515, glut_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
        else:
            subfig_loc_gad.draw_scatter(gad_type_data['x_zbrain'].astype(float), gad_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc_gad.draw_scatter(gad_type_data['z_zbrain'].astype(float) + 515, gad_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc_glut.draw_scatter(glut_type_data['x_zbrain'].astype(float), glut_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            subfig_loc_glut.draw_scatter(glut_type_data['z_zbrain'].astype(float) + 515, glut_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)

    subfig_loc_gad.draw_text(500, 820, 'gad')
    subfig_loc_gad.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_loc_gad.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')
    subfig_loc_glut.draw_text(500, 820, 'vglut')
    subfig_loc_glut.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
    subfig_loc_glut.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')

    for type, color, direction, supfig_gad, supfig_glut in zip(
            ['mr', 'lr', 'dkr', 'br', 'drr', 'dfr', 'ml', 'll', 'dkl', 'bl', 'drl', 'dfl',],
            ['#359B73', '#E69F00', '#9F0162',  '#F748A5', '#2271B2', '#D55E00', '#359B73', '#E69F00', '#9F0162',  '#F748A5', '#2271B2', '#D55E00', ],
            ['r', 'r', 'r', 'r', 'r', 'r', 'l', 'l', 'l', 'l', 'l', 'l', ],
            np.concatenate([subfigs_locs_gad, subfigs_locs_gad]),
            np.concatenate([subfigs_locs_Glut, subfigs_locs_Glut])):
        gad_type_data = gad_data[gad_data[type] == 1]
        glut_type_data = glut_data[glut_data[type] == 1]
        if direction == 'l':
            supfig_gad.draw_scatter(gad_type_data['x_zbrain'].astype(float), gad_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            supfig_gad.draw_scatter(gad_type_data['z_zbrain'].astype(float) + 515, gad_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            supfig_glut.draw_scatter(glut_type_data['x_zbrain'].astype(float), glut_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
            supfig_glut.draw_scatter(glut_type_data['z_zbrain'].astype(float) + 515, glut_type_data['y_zbrain'].astype(float), pc=color, ec=color, elw=0.25, ps=0.75, alpha=0.75)
        else:
            supfig_gad.draw_scatter(gad_type_data['x_zbrain'].astype(float), gad_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            supfig_gad.draw_scatter(gad_type_data['z_zbrain'].astype(float) + 515, gad_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            supfig_glut.draw_scatter(glut_type_data['x_zbrain'].astype(float), glut_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)
            supfig_glut.draw_scatter(glut_type_data['z_zbrain'].astype(float) + 515, glut_type_data['y_zbrain'].astype(float), pc='w', ec=color, ps=0.75, elw=0.25, alpha=0.75)

        supfig_gad.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        supfig_gad.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')
        supfig_glut.draw_line([35, 475, 475, 35, 35], [845, 845, 85, 85, 845], lc='r')
        supfig_glut.draw_line([545, 795, 795, 545, 545], [845, 845, 85, 85, 845], lc='r')


    for type_l, type_r, subfigs, color, fillcolor in zip(['ml', 'll', 'dkl', 'bl', 'drl', 'dfl', ],
                                                         ['mr', 'lr', 'dkr', 'br', 'drr', 'dfr', ],
                                                         subfigs_traces_gad,
                                                         ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                         ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']
                                                        ):
        print(f'gad: {type_l} - {type_r}')
        type_data = gad_data[gad_data[type_l] == 1]
        flip_stims = np.zeros(len(type_data))
        type_data = pd.concat([type_data, gad_data[gad_data[type_r] == 1]])
        flip_stims = np.concatenate([flip_stims, np.ones(len(gad_data[gad_data[type_r] == 1]))])

        traces = get_avg_traces_per_neuron(type_data['file_path'], type_data['cell_name_x'], type_data['z_plane'], flip_stims)

        for subfig, trace, stim in zip(subfigs, traces, ['lumi_left_dots_left',  'lumi_left_dots_right',  'lumi_left_dots_off',
                                                          'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                                          'lumi_off_dots_left',   'lumi_off_dots_right',   'lumi_off_dots_off']):

            subfig.draw_line(np.arange(0, 60, 0.5), np.nanpercentile(trace, 50, axis=0),
                             yerr_neg=np.nanpercentile(trace, 50, axis=0) - np.nanpercentile(trace, 25, axis=0),
                             yerr_pos=np.nanpercentile(trace, 75, axis=0) - np.nanpercentile(trace, 50),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')
        subfigs[6].draw_text(-20, 0., '0.5 dF/F\u2080', textlabel_rotation=90, textlabel_va="bottom")

    subfigs_traces_gad[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigs_traces_gad[4][8].draw_text(50, -0.6, '20s')

    for type_l, type_r, subfigs, color, fillcolor in zip(['ml', 'll', 'dkl', 'bl', 'drl', 'dfl', ],
                                                         ['mr', 'lr', 'dkr', 'br', 'drr', 'dfr', ],
                                                         subfigs_traces_Glut,
                                                         ['#359B73', '#E69F00', '#9F0162', '#F748A5', '#2271B2', '#D55E00'],
                                                         ['#8DCDB4', '#F7D280', '#CC7CAD', '#F7A4D0', '#93BADA', '#EEAE7C']
                                                         ):
        print(f'vglut: {type_l} - {type_r}')
        type_data = glut_data[glut_data[type_l] == 1]
        flip_stims = np.zeros(len(type_data))
        type_data = pd.concat([type_data, glut_data[glut_data[type_r] == 1]])
        flip_stims = np.concatenate([flip_stims, np.ones(len(glut_data[glut_data[type_r] == 1]))])

        traces = get_avg_traces_per_neuron(type_data['file_path'], type_data['cell_name_x'], type_data['z_plane'],
                                           flip_stims)

        for subfig, trace, stim in zip(subfigs, traces,
                                       ['lumi_left_dots_left', 'lumi_left_dots_right', 'lumi_left_dots_off',
                                        'lumi_right_dots_left', 'lumi_right_dots_right', 'lumi_right_dots_off',
                                        'lumi_off_dots_left', 'lumi_off_dots_right', 'lumi_off_dots_off']):
            subfig.draw_line(np.arange(0, 60, 0.5), np.nanpercentile(trace, 50, axis=0),
                             yerr_neg=np.nanpercentile(trace, 50, axis=0) - np.nanpercentile(trace, 25, axis=0),
                             yerr_pos=np.nanpercentile(trace, 75, axis=0) - np.nanpercentile(trace, 50),
                             lc=color, lw=1, eafc=fillcolor, eaalpha=1.0, ealw=1, eaec=fillcolor)

        subfigs[6].draw_line([-4, -4], [0, 0.5], lc='k')
        subfigs[6].draw_text(-20, 0., '0.5 dF/F\u2080', textlabel_rotation=90, textlabel_va="bottom")

    subfigs_traces_Glut[4][8].draw_line([40, 60], [-0.34, -0.34], lc='k')
    subfigs_traces_Glut[4][8].draw_text(50, -0.6, '20s')
    return

if __name__ == '__main__':
    hcr_fig = Figure(fig_width=9, fig_height=7)
    hcr_supfig = Figure(fig_width=18, fig_height=12)

    example_GC_path_HCR = r'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250408L\volume.nrrd'
    example_gad_path_HCR = r'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250408L\gadregistered2volume.nrrd'
    example_Glut_path_HCR = r'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250408L\vglutregistered2volume.nrrd'
    example_plane = 16
    example_neuron_data_path = fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250408L'
    example_neuron_file_base_name = '2025-04-08_10-20-26_fish000_setup0_arena0_functional'
    HCR_csv_overview_path = fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\all_cells_with_zbrain.csv'
    HCR_data_paths = [fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250331',
                      fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20241112-1',
                      fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250120left\left',
                      fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250217_1right',
                      fr'W:\M11 2P mircroscopes\Sophie\ExpWithKatja\20250408L']

    HCR_file_names = ['2025-03-31_11-46-00_fish000_setup0_arena0_functional',
                      '2024-11-12_11-40-51_fish000_SA',
                      '2025-01-20_10-47-06_fish000_setup0_arena0_SA',
                      '2025-02-17_18-40-48_fish001_setup1_arena0_functional',
                      '2025-04-08_10-20-26_fish000_setup0_arena0_functional']

    example_stack_GC_plot_HCR = hcr_fig.create_plot(xpos=0.25, ypos=3.8, plot_height=1.4, plot_width=1.4, axis_off=True,
                                                xmin=0, xmax=1150, ymin=1150, ymax=0)
    example_stack_gad_vglut_plot_HCR = hcr_fig.create_plot(xpos=3.5, ypos=3.8, plot_height=1.4, plot_width=1.4, axis_off=True,
                                                 xmin=0, xmax=1150, ymin=1150, ymax=0)

    example_loc_plot_overlap = hcr_fig.create_plot(xpos=5.5, ypos=3.8, plot_height=2.6, plot_width=2.6, axis_off=True)
    example_loc_plot_GC = hcr_fig.create_plot(xpos=8.15, ypos=3.8, plot_height=0.75, plot_width=0.75, axis_off=True)
    example_loc_plot_gad = hcr_fig.create_plot(xpos=8.15, ypos=4.725, plot_height=0.75, plot_width=0.75, axis_off=True)
    example_loc_plot_Glut = hcr_fig.create_plot(xpos=8.15, ypos=5.65, plot_height=0.75, plot_width=0.75, axis_off=True)

    example_neuron_plot_insitu = hcr_fig.create_plot(xpos=0.25, ypos=2.6, plot_height=0.75, plot_width=0.75, axis_off=True)
    example_neuron_plot_ref1 = hcr_fig.create_plot(xpos=1.4, ypos=2.6, plot_height=0.75, plot_width=0.75, axis_off=True)
    example_neuron_plot_ref2 = hcr_fig.create_plot(xpos=2.3, ypos=2.6, plot_height=0.75, plot_width=0.75, axis_off=True)
    example_neuron_traces_plots = create_traces_simple_subplots(hcr_fig, x_l=0.8, y_t=1.7, x_ss=0.75, y_ss=0.45, ymax_extra=-0.3)

    HCR_plot_ratios = hcr_fig.create_plot(xpos=4.5, ypos=1.3, xmin=-1, xmax=6, ymin=-0.05, ymax=1.05,
                               plot_height=2.1, plot_width=2.7, xticks=[0, 1, 2, 3, 4, 5], xticklabels=['motion', 'multifeature', 'luminance', 'change', 'increase', 'decrease'],
                               xticklabels_rotation=45, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yl='vglut\n(gad+vglut)')
    HCR_loc_gad = hcr_fig.create_plot(xpos=7.4, ypos=0.55, plot_height=1.5, plot_width=1.5, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=16.2-2.75, legend_ypos=14+2.75)
    HCR_loc_Glut = hcr_fig.create_plot(xpos=7.4, ypos=2.1, plot_height=1.5, plot_width=1.5, axis_off=True,
                                          xmin=30, xmax=800, ymin=850, ymax=80,
                                       legend_xpos=16.2-2.75, legend_ypos=14+2.75)

    HCR_subfigs_locs_gad = create_locs_subplots(hcr_supfig, x_l=1, y_t=5.85, x_bs=5.9, y_bs=5,)
    HCR_subfigs_locs_Glut = create_locs_subplots(hcr_supfig, x_l=3.9, y_t=5.85, x_bs=5.9, y_bs=5)

    HCR_subfigs_traces_gad = create_traces_subplots(hcr_supfig, x_l=0.75, y_t=10., x_bs=5.9, y_bs=5, x_ss=0.85)
    HCR_subfigs_traces_Glut = create_traces_subplots(hcr_supfig, x_l=3.65, y_t=10., x_bs=5.9, y_bs=5, x_ss=0.85)

    subfig_HCR_example_stacks(example_GC_path_HCR, example_gad_path_HCR, example_Glut_path_HCR,
                              example_stack_GC_plot_HCR, example_stack_gad_vglut_plot_HCR,
                              example_loc_plot_overlap, example_loc_plot_GC, example_loc_plot_gad, example_loc_plot_Glut,
                              example_plane)
    cell_name, cell_x_v, cell_y_v, z_plane_v, cell_x, cell_y, z_plane, tile = [10438,	124.235,	70.0626,	28,	414.34177,	239.20886,	0,	0]

    sub_plot_example_neuron(example_neuron_data_path, example_neuron_file_base_name,
                            example_neuron_plot_insitu, example_neuron_plot_ref1, example_neuron_plot_ref2, example_neuron_traces_plots,
                            cell_name=int(cell_name), cell_x_v=cell_x_v, cell_y_v=cell_y_v, z_plane_v=z_plane_v, cell_x=cell_x, cell_y=cell_y,
                            z_plane=z_plane, tile=tile)

    sub_plot_HCR_ratios(HCR_data_paths, HCR_file_names, HCR_csv_overview_path, HCR_plot_ratios, HCR_loc_gad, HCR_loc_Glut,
                       HCR_subfigs_traces_gad, HCR_subfigs_traces_Glut, HCR_subfigs_locs_gad, HCR_subfigs_locs_Glut)

    hcr_fig.save('C:/users/katja/Desktop/fig4.pdf')
    hcr_supfig.save('C:/users/katja/Desktop/subfig4.pdf')