import json
import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Turbofan RUL Predictive Maintenance AI', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 22px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #ffffff, #38bdf8, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0px;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2px;
    }

    .status-badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 12px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .status-badge-yellow {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 5px 12px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    .status-badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 5px 12px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
''', unsafe_allow_html=True)

from src.config import MODELS_FINAL_DIR, TRAIN_FILE, TEST_FILE, RUL_FILE, RAW_COLUMNS
from src.data.data_loader import load_official_test_data, load_official_rul_truth
from src.inference.predict import RULPredictor

@st.cache_data(show_spinner=False)
def get_cached_test_data():
    return load_official_test_data(TEST_FILE)

@st.cache_data(show_spinner=False)
def get_cached_rul_truth():
    return load_official_rul_truth(RUL_FILE)

@st.cache_resource(show_spinner=False)
def get_cached_predictor():
    try:
        return RULPredictor(MODELS_FINAL_DIR)
    except Exception:
        return None

st.markdown('''
<div class="main-header">
    <h1 class="main-title">NASA Turbofan AI: Remaining Useful Life (RUL) Predictor</h1>
    <p class="sub-title">Zero-Leakage Predictive Maintenance System - NASA C-MAPSS FD001</p>
</div>
''', unsafe_allow_html=True)

st.sidebar.title('System Status')
st.sidebar.success('**Champion Model**\n\n• Architecture: **Stacking Ensemble**\n• Test RMSE: **12.90 cycles**\n• Test MAE: **9.49 cycles**\n• R2 Score: **0.9036**\n• NASA Score: **220.79**')
st.sidebar.markdown('---')
st.sidebar.info('**Dataset Overview**\n\n• NASA C-MAPSS FD001\n• 21 Sensors & 3 Operating Settings\n• 100 Run-to-Failure Engines')

predictor = get_cached_predictor()
test_df = get_cached_test_data()
rul_truth = get_cached_rul_truth()

if predictor is None:
    st.error('Model artifacts not found in models/final/. Please train the pipeline first by running python3 main.py.')
else:
    input_tab1, input_tab2, input_tab3 = st.tabs(['Fleet Engine Selector', 'Custom Sensor Input Form', 'Upload Telemetry File'])

    input_df = None
    selected_cycle = 1
    actual_current_rul = None
    engine_id_display = 1

    with input_tab1:
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            engine_ids = sorted(test_df['engine_id'].unique().tolist())
            selected_engine = st.selectbox('Select Engine ID from Fleet:', engine_ids, index=14)
            engine_data = test_df[test_df['engine_id'] == selected_engine].sort_values('cycle').reset_index(drop=True)
            max_avail_cycle = len(engine_data)
            selected_cycle = st.slider('Simulate Flight Cycle Progression:', min_value=1, max_value=max_avail_cycle, value=max_avail_cycle)
        with col_ctrl2:
            st.info(f'**Engine #{selected_engine} History**: {max_avail_cycle} total cycles recorded. Inspecting telemetry slice up to **Cycle #{selected_cycle}**.')

        input_df = engine_data[engine_data['cycle'] <= selected_cycle].copy()
        engine_id_display = selected_engine
        true_final_rul = float(rul_truth[rul_truth['engine_id'] == selected_engine]['RUL'].values[0])
        actual_current_rul = true_final_rul + (max_avail_cycle - selected_cycle)

    with input_tab2:
        st.markdown('##### Quick State Presets')
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            if st.button('Healthy State (Early Life)', use_container_width=True):
                st.session_state['cycle_in'] = 30
                st.session_state['s2_in'] = 642.15
                st.session_state['s3_in'] = 1582.40
                st.session_state['s4_in'] = 1400.20
                st.session_state['s7_in'] = 554.20
                st.session_state['s8_in'] = 2388.02
                st.session_state['s9_in'] = 9050.10
                st.session_state['s11_in'] = 47.15
                st.session_state['s12_in'] = 522.40
                st.session_state['s15_in'] = 8.38
                st.session_state['s20_in'] = 39.02
                st.session_state['s21_in'] = 23.42
                st.rerun()
        with pcol2:
            if st.button('Warning State (Degrading)', use_container_width=True):
                st.session_state['cycle_in'] = 140
                st.session_state['s2_in'] = 642.85
                st.session_state['s3_in'] = 1592.10
                st.session_state['s4_in'] = 1412.50
                st.session_state['s7_in'] = 553.10
                st.session_state['s8_in'] = 2388.10
                st.session_state['s9_in'] = 9068.50
                st.session_state['s11_in'] = 47.62
                st.session_state['s12_in'] = 521.15
                st.session_state['s15_in'] = 8.46
                st.session_state['s20_in'] = 38.75
                st.session_state['s21_in'] = 23.25
                st.rerun()
        with pcol3:
            if st.button('Critical State (Imminent Wear)', use_container_width=True):
                st.session_state['cycle_in'] = 220
                st.session_state['s2_in'] = 643.80
                st.session_state['s3_in'] = 1601.50
                st.session_state['s4_in'] = 1426.00
                st.session_state['s7_in'] = 551.80
                st.session_state['s8_in'] = 2388.22
                st.session_state['s9_in'] = 9085.40
                st.session_state['s11_in'] = 48.15
                st.session_state['s12_in'] = 519.80
                st.session_state['s15_in'] = 8.56
                st.session_state['s20_in'] = 38.35
                st.session_state['s21_in'] = 23.05
                st.rerun()

        with st.form('custom_input_form'):
            st.markdown('##### Sensor Parameters')
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                man_engine_id = st.number_input('Engine ID', min_value=1, max_value=9999, value=1, step=1)
                man_cycle = st.number_input('Cycle Number', min_value=1, max_value=800, value=st.session_state.get('cycle_in', 60), step=1)
            with c2:
                man_s2 = st.number_input('Sensor 2: LPC Outlet Temp (T24)', value=st.session_state.get('s2_in', 642.50), format='%.2f')
                man_s3 = st.number_input('Sensor 3: HPC Outlet Temp (T30)', value=st.session_state.get('s3_in', 1589.00), format='%.2f')
            with c3:
                man_s4 = st.number_input('Sensor 4: LPT Outlet Temp (T50)', value=st.session_state.get('s4_in', 1406.00), format='%.2f')
                man_s7 = st.number_input('Sensor 7: HPC Outlet Press (P30)', value=st.session_state.get('s7_in', 553.50), format='%.2f')
            with c4:
                man_s11 = st.number_input('Sensor 11: Static Press (Ps30)', value=st.session_state.get('s11_in', 47.45), format='%.2f')
                man_s15 = st.number_input('Sensor 15: Bypass Ratio (BPR)', value=st.session_state.get('s15_in', 8.43), format='%.4f')

            c_sub1, c_sub2, c_sub3, c_sub4 = st.columns(4)
            with c_sub1:
                man_s8 = st.number_input('Sensor 8: Fan Speed (Nf)', value=st.session_state.get('s8_in', 2388.08), format='%.2f')
            with c_sub2:
                man_s9 = st.number_input('Sensor 9: Core Speed (Nc)', value=st.session_state.get('s9_in', 9060.00), format='%.2f')
            with c_sub3:
                man_s20 = st.number_input('Sensor 20: HPT Coolant Bleed', value=st.session_state.get('s20_in', 38.85), format='%.2f')
            with c_sub4:
                man_s21 = st.number_input('Sensor 21: LPT Coolant Bleed', value=st.session_state.get('s21_in', 23.32), format='%.2f')

            run_manual_btn = st.form_submit_button('Run RUL Prediction', use_container_width=True)

        if run_manual_btn:
            manual_record = {
                'engine_id': man_engine_id, 'cycle': man_cycle, 'setting_1': 0.0, 'setting_2': 0.0, 'setting_3': 100.0,
                'sensor_1': 518.67, 'sensor_2': man_s2, 'sensor_3': man_s3, 'sensor_4': man_s4, 'sensor_5': 14.62,
                'sensor_6': 21.61, 'sensor_7': man_s7, 'sensor_8': man_s8, 'sensor_9': man_s9, 'sensor_10': 1.30,
                'sensor_11': man_s11, 'sensor_12': st.session_state.get('s12_in', 521.60), 'sensor_13': 2388.08,
                'sensor_14': 8140.00, 'sensor_15': man_s15, 'sensor_16': 0.03, 'sensor_17': 392.0, 'sensor_18': 2388.0,
                'sensor_19': 100.0, 'sensor_20': man_s20, 'sensor_21': man_s21
            }
            input_df = pd.DataFrame([manual_record])
            selected_cycle = man_cycle
            engine_id_display = man_engine_id
            actual_current_rul = None

    with input_tab3:
        st.markdown('##### Upload Telemetry Log (CSV, TXT, or Excel)')
        uploaded_file = st.file_uploader('Upload engine flight cycles:', type=['csv', 'txt', 'xlsx', 'xls', 'tsv'])
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(('.xlsx', '.xls')):
                    up_df = pd.read_excel(uploaded_file)
                elif fname.endswith('.tsv'):
                    up_df = pd.read_csv(uploaded_file, sep='\t')
                else:
                    try:
                        up_df = pd.read_csv(uploaded_file)
                        if up_df.shape[1] == 1:
                            uploaded_file.seek(0)
                            up_df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
                    except Exception:
                        uploaded_file.seek(0)
                        up_df = pd.read_csv(uploaded_file, sep=r'\s+', header=None)

                if up_df.shape[1] >= 26 and 'sensor_1' not in up_df.columns:
                    up_df = up_df.iloc[:, :26]
                    up_df.columns = RAW_COLUMNS
                elif up_df.shape[1] == 24 and 'sensor_1' not in up_df.columns:
                    col_names = [c for c in RAW_COLUMNS if c not in ['engine_id', 'cycle']]
                    up_df.columns = col_names
                    up_df['engine_id'] = 1
                    up_df['cycle'] = list(range(1, len(up_df) + 1))
                elif 'engine_id' not in up_df.columns:
                    up_df['engine_id'] = 1
                    if 'cycle' not in up_df.columns:
                        up_df['cycle'] = list(range(1, len(up_df) + 1))

                st.success(f'Loaded {len(up_df)} cycles across {up_df["engine_id"].nunique()} engine(s).')
                
                avail_up_eng = sorted(up_df['engine_id'].unique().tolist())
                if len(avail_up_eng) > 1:
                    chosen_up_eng = st.selectbox('Select Engine from Uploaded File:', avail_up_eng)
                    chosen_slice = up_df[up_df['engine_id'] == chosen_up_eng].copy().reset_index(drop=True)
                else:
                    chosen_slice = up_df.copy().reset_index(drop=True)

                st.dataframe(chosen_slice.head(5), use_container_width=True)
                input_df = chosen_slice
                selected_cycle = int(chosen_slice['cycle'].max()) if 'cycle' in chosen_slice.columns else len(chosen_slice)
                engine_id_display = int(chosen_slice['engine_id'].iloc[-1]) if 'engine_id' in chosen_slice.columns else 1
                actual_current_rul = None
            except Exception as e:
                st.error(f'Error reading file: {e}')

    if input_df is not None:
        inf_result = predictor.predict_engine(input_df)
        pred_rul = inf_result['predicted_rul_cycles']
        est_failure = inf_result['estimated_failure_cycle']
        status_text = inf_result['health_status']
        status_color = inf_result['status_indicator']

        st.markdown('---')
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Current Cycle</div>
                <div class="metric-value">{selected_cycle}</div>
            </div>
            ''', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Predicted RUL</div>
                <div class="metric-value" style="color: #38bdf8;">{pred_rul:.1f} <span style="font-size: 0.95rem; color:#94a3b8;">cycles</span></div>
            </div>
            ''', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Estimated Failure</div>
                <div class="metric-value">Cycle {est_failure:.0f}</div>
            </div>
            ''', unsafe_allow_html=True)
        with mcol4:
            if actual_current_rul is not None:
                abs_error = abs(pred_rul - actual_current_rul)
                err_color = '#10b981' if abs_error <= 15 else '#f59e0b'
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">True RUL & Error</div>
                    <div class="metric-value" style="color: {err_color};">{actual_current_rul:.0f} <span style="font-size: 0.9rem; color:#94a3b8;">(±{abs_error:.1f})</span></div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">Operational Margin</div>
                    <div class="metric-value" style="color: #38bdf8;">{pred_rul:.1f} <span style="font-size: 0.9rem; color:#94a3b8;">cycles</span></div>
                </div>
                ''', unsafe_allow_html=True)
        with mcol5:
            badge_class = f'status-badge-{status_color.lower()}'
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Health Status</div>
                <div style="margin-top: 6px;">
                    <span class="{badge_class}">{status_color}: {status_text.split('(')[0].strip()}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)

        fig_rul = go.Figure()
        if len(input_df) > 1 and 'cycle' in input_df.columns:
            past_cycles = input_df['cycle'].values
            y_prog = [pred_rul + (selected_cycle - c) for c in past_cycles]
            fig_rul.add_trace(go.Scatter(
                x=past_cycles, y=y_prog, mode='lines', name='Observed Trajectory', line=dict(color='#38bdf8', width=3)
            ))
        else:
            fig_rul.add_trace(go.Scatter(
                x=[selected_cycle], y=[pred_rul], mode='markers', name='Current Observation', marker=dict(size=10, color='#38bdf8')
            ))

        fig_rul.add_trace(go.Scatter(
            x=[selected_cycle, est_failure], y=[pred_rul, 0], mode='lines', name='Forecast Path', line=dict(color='#ef4444', width=2.5, dash='dash')
        ))
        fig_rul.add_trace(go.Scatter(
            x=[est_failure], y=[0], mode='markers', marker=dict(size=13, color='#ef4444', symbol='x'), name='Failure Limit', hovertext=f'Failure at Cycle {est_failure:.0f}'
        ))

        if actual_current_rul is not None:
            fig_rul.add_trace(go.Scatter(
                x=[selected_cycle], y=[actual_current_rul], mode='markers', marker=dict(size=11, color='#10b981', symbol='diamond'), name='Actual Ground Truth RUL', hovertext=f'True RUL: {actual_current_rul:.0f}'
            ))

        fig_rul.add_vline(x=est_failure, line_dash='dot', line_color='rgba(239, 68, 68, 0.6)', line_width=1.5)

        fig_rul.add_hrect(y0=0, y1=15, fillcolor='rgba(239, 68, 68, 0.15)', line_width=0, annotation_text='CRITICAL (RUL <= 15)', annotation_position='top left', annotation_font_size=10, annotation_font_color='#f87171')
        fig_rul.add_hrect(y0=15, y1=40, fillcolor='rgba(245, 158, 11, 0.10)', line_width=0, annotation_text='WARNING (15 < RUL <= 40)', annotation_position='top left', annotation_font_size=10, annotation_font_color='#fbbf24')
        fig_rul.add_hrect(y0=40, y1=240, fillcolor='rgba(16, 185, 129, 0.07)', line_width=0, annotation_text='HEALTHY OPERATING MARGIN', annotation_position='top left', annotation_font_size=10, annotation_font_color='#34d399')

        fig_rul.update_layout(
            title=dict(text=f'<b>Engine #{engine_id_display}: Degradation & RUL Forecast</b>', y=0.96),
            xaxis_title='Flight Cycles',
            yaxis_title='Remaining Useful Life (Cycles)',
            yaxis=dict(range=[-10, max(pred_rul + selected_cycle + 25, 160)]),
            template='plotly_dark',
            height=430,
            margin=dict(l=20, r=20, t=50, b=65),
            legend=dict(orientation='h', yanchor='top', y=-0.22, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_rul, use_container_width=True, config={'displayModeBar': False})

        with st.expander('View Input Telemetry Table'):
            st.dataframe(input_df.tail(20), use_container_width=True)
