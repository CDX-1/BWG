[SENSOR-LOSS §1.1]
Loss of an attitude sensor (IMU, star tracker, sun sensor) without confirmed
replacement invalidates every downstream pointing decision. Do NOT command
attitude changes on stale sensor data — a frozen sensor reading looks like
"perfectly stable" and can mask an actual tumble. The first response to any
suspected sensor loss is to freeze the attitude estimate at the last valid
sample and stop trusting live input from the suspect stream.

[SENSOR-LOSS §1.2]
Historical precedent — Hubble Space Telescope suffered progressive gyroscope
failures across its lifetime. When only 3 of 6 gyros remained functional,
the mission was extended by entering "reduced gyro mode" that used 2 gyros
plus the fine guidance star trackers to reconstruct attitude. The pattern
is generalisable: sensor loss recovery often requires COMPOSING surviving
sensors that were not originally intended to substitute for the failed one.

[SENSOR-LOSS §2.1]
If the primary IMU shows stale (unchanging) values while the payload data
link remains active, the sensor is frozen but the bus is alive. Preserve
the last known attitude estimate and switch to a redundant sensor. Star
tracker is preferred over sun sensor for fine pointing; sun sensor is
preferred for coarse safe-mode acquisition.

[SENSOR-LOSS §2.2]
Historical precedent — Kepler space telescope lost its second reaction
wheel in 2013, ending the nominal mission. The K2 recovery used solar
radiation pressure as a "virtual third wheel", stabilising the spacecraft
against an axis that wasn't intended to be balanced this way. The lesson:
when the obvious recovery path is unavailable, previously ignored physics
often provides one.

[SENSOR-LOSS §3.1]
Ultrasonic ranging (used for proximity and hazard detection) has intrinsic
ambiguity: a "no echo" return may mean either (a) no target in the field
of view, or (b) transducer failure. Sustained no-echo greater than 5 s
during a maneuver window with an expected target warrants a transducer
diagnostic. Sustained no-echo outside a maneuver window is not by itself
evidence of failure.

[SENSOR-LOSS §3.2]
Historical precedent — Cassini reaction wheel bearing degradation was
detected via a slow increase in the torque command needed to maintain the
same wheel speed. Trend analysis of a SECONDARY indicator caught the
failure signature before the sensor itself dropped out. Sensor-loss
detection should watch both the sensor's own output AND the derived
quantities that depend on it.

[SENSOR-LOSS §4.1]
When multiple sensors fail simultaneously, treat as a bus or power fault
before assuming sensor-side failure. Sensor loss following a power event
is likely a downstream effect; sensor loss without a power event and with
the bus still delivering frames is likely a sensor fault. The recovery
priority order changes accordingly — a bus fault demands a bus recovery,
not a per-sensor switch.

[SENSOR-LOSS §4.2]
Historical precedent — Mars Reconnaissance Orbiter (MRO) has operated for
over a decade with progressive IMU degradation. Its "IMU-off" cruise mode
uses attitude propagation from star tracker sightings interleaved with
brief IMU wake-ups, extending sensor lifetime by an order of magnitude.
Dead-reckoning WITH periodic sensor rechecks is more sustainable than
either full reliance on a suspect sensor or a hard switch to backup.

[SENSOR-LOSS §5.1]
Recovery priority order for a suspected sensor loss event:
1. Preserve last known state — freeze the attitude estimate at the last
   sample that was demonstrably valid.
2. Switch to backup sensor if one is available and has recent validation.
3. Enable dead reckoning using the orbital model, WITH periodic sensor
   rechecks so drift stays bounded.
4. Enter safe mode with sun-pointing if no backup sensor is available and
   dead reckoning cannot be closed within the mission's pointing tolerance.
5. Request ground intervention for cold reset or hardware reconfiguration
   only when steps 1-4 are exhausted, because Earth round-trip delay makes
   this the slowest possible recovery path.

[SENSOR-LOSS §5.2]
When an IMU loss is confirmed on a spacecraft that also has a compromised
communications path, DO NOT enter safe mode by default — safe mode
typically re-points the vehicle, and re-pointing without live attitude
sensing is a compound risk. Prefer attitude hold with the last known
estimate until either sensing or comms is restored.

[SENSOR-LOSS §6.1]
Evidence preservation during a sensor loss event is disproportionately
important because the sensor itself cannot be replayed. Prioritise for
downlink: the sensor's own telemetry immediately before and after the
loss event (raw, not summarised), the bus status log confirming whether
frames were still arriving, and any subsystem log that recorded a state
transition attributable to the sensor loss. Loss of THIS evidence
permanently removes ground's ability to determine whether the sensor
failed, the bus failed, or the software misinterpreted a valid reading.
