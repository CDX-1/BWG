[SPECTROMETER-MANUAL §1.1]
The Asteria-7 spectrometer operates in continuous sampling mode during science passes.
Nominal baseline output is 10.0 ± 0.5 counts. Values exceeding 14.0 are flagged for review.
Peak readings above 18.0 indicate a significant transient event requiring investigation.

[SPECTROMETER-MANUAL §2.3]
The spectrometer's analog-to-digital converter is susceptible to single-event upsets (SEUs)
caused by energetic particle strikes. An SEU may produce a spurious spike indistinguishable
from a real spectral event without corroborating radiation data.

[SPECTROMETER-MANUAL §3.2]
Short transient signals must not be temporally averaged before review.
Averaging may remove asymmetry, peak structure, or secondary pulses.
Raw frames spanning at least T-3s to T+3s must be preserved for any event under investigation.

[SPECTROMETER-MANUAL §4.1]
Cross-correlation between spectrometer output and radiation counters is required to distinguish
external particle events from instrument electronics faults. Both raw streams must be available
for investigators to perform this analysis.

[SPECTROMETER-MANUAL §5.7]
Temperature drift in the spectrometer detector array can produce gain shifts of up to 0.3 counts
per degree Celsius. A transient combined with thermal drift requires both thermal telemetry and
raw spectrum data for accurate interpretation.
