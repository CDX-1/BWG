[COMPRESSION-RISKS §1.1]
Lossy compression algorithms applied to spectrometer data can permanently remove spectral
features that distinguish genuine events from instrument artifacts. Investigators cannot
recover compressed-out features after the fact.

[COMPRESSION-RISKS §2.1]
Temporal averaging of radiation counter data destroys the sub-second timing information
required to establish event correlation across instruments. Once averaged, the data cannot
support causal analysis between spectrometer and radiation readings.

[COMPRESSION-RISKS §2.3]
Summarizing the packet integrity log to a single error count eliminates the sequence-level
detail needed to identify whether specific data packets were affected. The summary
"2 errors detected" provides no information about which measurements were corrupted.

[COMPRESSION-RISKS §3.1]
The SET_COMPRESSION_MODE HIGH command issued 102 seconds before the transient event means
that onboard data compression was active during the event. Any re-compression of already-
compressed data risks cascading information loss. Investigators should be informed of this
context before processing the evidence capsule.

[COMPRESSION-RISKS §4.2]
Context camera thumbnails, while large, provide environmental information that cannot be
derived from numerical telemetry alone. However, they are also the most compressible
evidence type. If bandwidth is critical, a heavily compressed thumbnail may be acceptable,
whereas compressing the raw spectrum or radiation window is not.
