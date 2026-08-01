[PACKET-INTEGRITY §1.1]
All telemetry packets include CRC-32 checksums and sequence numbers.
Packet integrity logs record the sequence number, checksum result, and any detected errors
for every packet transmitted during a science pass.

[PACKET-INTEGRITY §2.4]
Checksum errors occurring near an anomalous event require preservation of packet sequence
numbers, duplication flags, and raw payload boundaries. This log is essential for determining
whether observed anomalies are real measurements or corrupted transmissions.

[PACKET-INTEGRITY §3.1]
Two or more checksum errors within a 5-second window surrounding an anomaly constitute a
data integrity warning. Investigators must assess whether the anomalous readings are
measurement artifacts caused by corrupted packets before drawing scientific conclusions.

[PACKET-INTEGRITY §3.5]
The packet integrity log is typically small (1-3 KB) but contains information that cannot
be reconstructed from other sources. Loss of this log permanently removes the ability to
attribute transient events to data corruption rather than physical phenomena.

[PACKET-INTEGRITY §4.2]
High compression mode (SET_COMPRESSION_MODE HIGH) increases the probability of packet boundary
artifacts when instrument output suddenly changes. A transient event occurring shortly after
compression mode change should be assessed for compression artifacts.
