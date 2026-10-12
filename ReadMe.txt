SD-IoV QoS Algorithm Comparison
This repository contains Python scripts, simulation datasets, raw result
files, and comparison tables for evaluating QoS-related performance in a
Software-Defined Internet of Vehicles (SD-IoV) environment.
The files are organized around three performance metrics:
- End-to-end latency
- Packet drop
- Flows failing QoS requirements
The repository includes files associated with LSTM, MaxWeight, and
Greedy comparisons, along with Lyapunov-related dataset and
controller-trace files. Please refer to the corresponding script and
dataset contents for the exact model configuration, equations, and
simulation parameters.
Repository contents
Python scripts
  File                                     Purpose
  avg_pkt_drop.py                        Python script related to average
                                           packet-drop analysis or
                                           calculation.
  bar.py                                 Python script for generating
                                           bar-chart visualizations.
  lstm_greedy_maxweight_latency.py       Script for the LSTM, Greedy, and
                                           MaxWeight latency comparison.
  qos_failure_lstm_maxweight_greedy.py   Script for comparing the percentage
                                           of flows that fail their QoS
                                           requirements across LSTM,
                                           MaxWeight, and Greedy.
  sdvn_algorithm_comparison.py           Script for SDVN algorithm
                                           comparison. Check the source for
                                           the specific metrics and
                                       configuration it uses.
Dataset and trace files
  File                                Purpose
  LyapunovDataSet.xlsx              Excel dataset associated with the
                                      Lyapunov-based study. See the
                                      workbook sheets and column headings
                                      for the exact fields.
  proposed_controller_trace.csv     Controller trace for the proposed
                                      approach; inspect the column
                                      headings for the recorded events
                                      and measurements.
  raw_runs.csv                      Raw run-level results used for
                                  analysis or comparison.
Latency results
  File                                Purpose
  latency_raw_results.csv           Raw latency measurements or
                                      run-level latency results.
  latency_comparison_table.csv      Processed latency comparison table.
Packet-drop results
  File                                 Purpose
  packet_drop_raw_results.csv        Raw packet-drop measurements or
                                       run-level results.
  packet_drop_comparison_table.csv   Processed packet-drop comparison
                                   table.
QoS-failure results
  File                                 Purpose
  qos_failure_raw_results.csv        Raw results related to flows that
                                       fail their QoS requirements.
  qos_failure_comparison_table.csv   Processed QoS-failure comparison
                                   table.
Summary
  File                                Purpose
  summary.csv                       Summary of results. Check the
                                      column names to identify the
                                      metrics, algorithms, and
                                      aggregation method included.