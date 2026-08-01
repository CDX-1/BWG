[INCIDENT-RESPONSE §1.1]
When an unclassified anomaly is detected during a science pass, the spacecraft must preserve
the highest-fidelity available records for ground investigation. Evidence lost during
transmission cannot be recovered — the spacecraft does not store original data indefinitely.

[INCIDENT-RESPONSE §2.1]
Priority preservation order for a spectrometer transient:
1. Raw spectrometer frames (primary instrument data)
2. Correlated sensor data (radiation, thermal) within the event window
3. Data integrity logs (packet checksums, sequence records)
4. Supporting context (power telemetry, camera thumbnails)

[INCIDENT-RESPONSE §2.3]
The evidence capsule must support at least three independent investigative paths:
(a) external environmental event analysis, (b) instrument electronics fault analysis,
and (c) data integrity assessment. Removing evidence that supports only one path
may be acceptable; removing evidence that supports all three is not.

[INCIDENT-RESPONSE §3.2]
A 25 KB transmission budget requires careful prioritization. Do not transmit pre-processed
or summarized versions of data if the raw version is within budget. Summaries produced
onboard may embed assumptions that are incorrect given the actual event nature.

[INCIDENT-RESPONSE §4.1]
Mission control requires an uncertainty statement confirming that all plausible explanations
remain viable. A capsule that supports only one explanation is insufficient even if that
explanation appears most likely to onboard systems.

[INCIDENT-RESPONSE §5.1]
Recommended follow-up for unresolved spectrometer transients: schedule a repeat science pass
during a quieter magnetospheric phase, with the spectrometer in low-compression mode and
all correlated instruments recording at maximum sample rate.
