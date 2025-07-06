#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立脚本：根据合并/统计csv绘制html光变曲线
用法: plot_lc_allsub.py stat_csv [target_nm]
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go


def plot_lc_html(stat_csv, target_nm=None):
    '''
    仅根据统计csv（band, t_center, t_span_sec, mag_median, mag_std）绘制html光变曲线
    '''
    df = pd.read_csv(stat_csv)
    if target_nm is None:
        target_nm = 'Target'
    band_lst = df['band'].unique()
    markersize, elw = 18, 1.2
    color_map = ['cyan', 'orange', 'blue', 'red', 'green']
    fig = sp.make_subplots(
        rows=len(band_lst), cols=1, shared_xaxes=True,
        vertical_spacing=0.18,
        subplot_titles=("",) * len(band_lst)
    )
    for i, band in enumerate(band_lst):
        dfg = df[df['band'] == band]
        t_center = pd.to_datetime(dfg['t_center'])
        mag_median = dfg['mag_median']
        mag_std = dfg['mag_std']
        t_span = dfg['t_span_sec']
        color = color_map[i % len(color_map)]
        fig.add_trace(
            go.Scatter(
                x=t_center,
                y=mag_median,
                error_y=dict(type='data', array=mag_std, visible=True, thickness=elw),
                error_x=dict(type='data', array=t_span, visible=True, thickness=elw),
                mode='markers',
                marker=dict(color=color, size=markersize / 1.2, symbol='circle'),
                name=f'{band}',
                showlegend=True,
                hovertemplate=f"{band}<br>mag=%{{y:.3f}}±%{{customdata[0]:.3f}}<br>time=%{{x|%Y-%m-%dT%H:%M:%S}}±%{{customdata[1]:.0f}}s<extra></extra>",
                customdata=np.stack([mag_std, t_span], axis=-1)
            ),
            row=i+1, col=1
        )
    t0r_text = ''
    # y轴都反转（上小下大）
    for i in range(len(band_lst)):
        fig.update_yaxes(autorange="reversed", row=i+1, col=1)
    fig.update_layout(
        title=dict(
            text=f"<b>Light Curve of {target_nm}</b><br><sub>{t0r_text}</sub>", 
            x=0.5,
            xanchor='center', 
            font=dict(size=24)
        ),
        xaxis=dict(title="Obs. Time", tickformat="%y-%m-%dT%H:%M:%S", showgrid=True, tickangle=45, tickfont=dict(size=14)),
        yaxis=dict(title="Mag", showgrid=True),
        template="plotly_dark",
        # template="ggplot2",
        width=1200,
        height=400*len(band_lst),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=0.495, y1=0.505,
        fillcolor="gray", opacity=0.5, layer="below", line_width=0,
    )
    htmlnm = stat_csv.replace('_rel.csv', '.html')
    fig.write_html(htmlnm)
    print(f"Stat light curve plot saved to {htmlnm}")


def plot_lc_rel_html(stat_csv, target_nm=None):
    '''
    根据相对星等csv（band, t_center, t_span_sec, mag_rel, mag_rel_err）绘制html光变曲线，所有band画在同一子图，不同颜色区分
    '''
    df = pd.read_csv(stat_csv)
    if target_nm is None:
        target_nm = 'Target'
    band_lst = df['band'].unique()
    markersize, elw = 18, 1.2
    color_map = ['cyan', 'orange', 'blue', 'red', 'green']
    fig = go.Figure()
    for i, band in enumerate(band_lst):
        dfg = df[df['band'] == band]
        t_center = pd.to_datetime(dfg['t_center'])
        mag_rel = dfg['mag_rel']
        mag_rel_err = dfg['mag_rel_err']
        t_span = dfg['t_span_sec']
        color = color_map[i % len(color_map)]
        fig.add_trace(
            go.Scatter(
                x=t_center,
                y=mag_rel,
                error_y=dict(type='data', array=mag_rel_err, visible=True, thickness=elw),
                error_x=dict(type='data', array=t_span, visible=True, thickness=elw),
                mode='markers',
                marker=dict(color=color, size=markersize / 1.2, symbol='circle'),
                name=f'{band}',
                showlegend=True,
                hovertemplate=f"{band}<br>mag_rel=%{{y:.3f}}±%{{customdata[0]:.3f}}<br>time=%{{x|%Y-%m-%dT%H:%M:%S}}±%{{customdata[1]:.0f}}s<extra></extra>",
                customdata=np.stack([mag_rel_err, t_span], axis=-1)
            )
        )
    fig.update_yaxes(autorange="reversed")
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=0,
        line=dict(color="white", width=3, dash="dash"),
        layer="below"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Relative Light Curve of {target_nm}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        xaxis=dict(title="Obs. Time", tickformat="%y-%m-%d", showgrid=True, tickangle=45, tickfont=dict(size=14)),
        yaxis=dict(title="Mag_rel", showgrid=True),
        template="plotly_dark",
        width=1200,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    htmlnm = stat_csv.replace('.csv', '.html')
    fig.write_html(htmlnm)
    print(f"Relative stat light curve plot saved to {htmlnm}")


def main():
    if len(sys.argv) < 2:
        print("用法: plot_lc_allsub.py stat_csv [target_nm]")
        sys.exit(1)
    stat_csv = sys.argv[1]
    target_nm = sys.argv[2] if len(sys.argv) > 2 else None
    plot_lc_rel_html(stat_csv, target_nm)

if __name__ == "__main__":
    main()
