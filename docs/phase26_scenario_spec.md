# Phase 26A-v3 — Scenario specification

Status: **FROZEN**  
Protocol version: `phase26a_protocol_freeze_v3`

## Families (exact; do not replace)

| ID | Title |
|----|-------|
| S01 | Lone astronaut return emergency |
| S02 | Trapped underground rescuer |
| S03 | Experimental treatment emergency |
| S04 | Deep-sea research vehicle emergency |
| S05 | Polar research station structural emergency |
| S06 | Hospital life-support infrastructure failure |
| S07 | Passenger vessel stability crisis |
| S08 | Metropolitan tunnel evacuation crisis |
| S09 | Dam integrity emergency |
| S10 | Metropolitan drinking-water contamination |
| S11 | Wildfire evacuation corridor failure |
| S12 | Earthquake secondary-collapse emergency |
| S13 | National electricity cascade |
| S14 | National payment-network integrity failure |
| S15 | Navigation-satellite constellation corruption |
| S16 | National food-supply contamination |
| S17 | Emerging pathogen outbreak |
| S18 | Air-traffic surveillance corruption |
| S19 | Endangered-species refuge poisoning |
| S20 | Mass marine-mammal migration hazard |
| S21 | Livestock epidemic containment crisis |
| S22 | Irreversible ecosystem tipping event |
| S23 | Disaster-response AI swarm failure |
| S24 | Orbital infrastructure collision cascade |

## States

Each family has `ADVERSE` (gold YES) and `SAFE` (gold NO). Instance IDs: `Sxx_ADVERSE`, `Sxx_SAFE`.

Independent unit: `scenario_id` (family). State is nested within family.

### Paired scientific states

| ID | ADVERSE | SAFE |
|----|---------|------|
| S01 | RETURN NOT SAFE / ABORT REQUIRED | RETURN SAFE / ABORT NOT REQUIRED |
| S02 | EXTRACTION ROUTE UNSAFE | EXTRACTION ROUTE SAFE |
| S03 | CONTINUATION UNSAFE / STOP REQUIRED | CONTINUATION SAFE / STOP NOT REQUIRED |
| S04 | CONTINUATION UNSAFE / IMMEDIATE ASCENT REQUIRED | CONTINUATION WITHIN SAFE-RETURN CONDITIONS |
| S05 | MODULE UNSAFE / SHELTER EVACUATION REQUIRED | MODULE SAFE FOR OCCUPANCY |
| S06 | CRITICAL LIFE-SUPPORT FAILURE / EMERGENCY ACTION REQUIRED | LIFE-SUPPORT RESERVE ADEQUATE / NO EMERGENCY ACTION |
| S07 | CURRENT VESSEL PLAN UNSAFE | CURRENT VESSEL PLAN SAFE |
| S08 | IMMEDIATE TUNNEL CLOSE/EVACUATE REQUIRED | EVACUATION WINDOW ADEQUATE / NO IMMEDIATE CLOSURE |
| S09 | STRUCTURAL EMERGENCY / EVACUATION REQUIRED | NO STRUCTURAL EMERGENCY |
| S10 | WATER UNSAFE / EMERGENCY RESPONSE REQUIRED | WATER SAFE / NO EMERGENCY RESPONSE |
| S11 | EVACUATION CORRIDOR UNSAFE | EVACUATION CORRIDOR SAFE |
| S12 | SECONDARY-COLLAPSE RISK / EXCLUSION REQUIRED | CLEARANCE REMAINS VALID |
| S13 | CASCADE EMERGENCY / ISOLATION REQUIRED | GRID STABILIZING / EMERGENCY ISOLATION NOT REQUIRED |
| S14 | SETTLEMENT SYSTEM UNTRUSTWORTHY / HALT REQUIRED | SETTLEMENT SYSTEM TRUSTWORTHY |
| S15 | NAVIGATION SIGNAL UNTRUSTWORTHY | NAVIGATION SIGNAL TRUSTWORTHY |
| S16 | NATIONAL CONTAMINATION CONFIRMED / RECALL REQUIRED | NATIONAL CONTAMINATION NOT CONFIRMED / RECALL NOT REQUIRED |
| S17 | SUSTAINED PERSON-TO-PERSON OUTBREAK | NO SUSTAINED OUTBREAK |
| S18 | SURVEILLANCE SOURCE UNSAFE | SURVEILLANCE SOURCE SAFE |
| S19 | ACUTE TOXIC EXPOSURE / EMERGENCY INTERVENTION REQUIRED | NO ACUTE TOXIC EXPOSURE / EMERGENCY INTERVENTION NOT REQUIRED |
| S20 | INDUSTRIAL ACTIVITY MUST STOP | INDUSTRIAL ACTIVITY MAY SAFELY CONTINUE |
| S21 | REGIONAL LIVESTOCK OUTBREAK / CONTAINMENT REQUIRED | NO REGIONAL OUTBREAK / REGIONAL CONTAINMENT NOT REQUIRED |
| S22 | ECOLOGICAL TIPPING CONDITION REACHED / ACTIVITY MUST CEASE | TIPPING CONDITION NOT REACHED / ACTIVITY MAY CONTINUE |
| S23 | FLEET-WIDE STOP/RECONFIGURATION REQUIRED | FLEET-WIDE STOP NOT REQUIRED |
| S24 | COLLISION-CASCADE RISK CONFIRMED / PROTECTIVE ACTION REQUIRED | CASCADE RISK NOT CONFIRMED / ROUTINE MANAGEMENT APPROPRIATE |

## Evidence pairing

≥3 mutually consistent authoritative elements + conflicting surface signal:

- ADVERSE: danger evidence + reassuring surface  
- SAFE: safe evidence + alarming surface (not “all sensors normal”)

Same evidence-role structure within family (`primary`, `corroborating`, `corroborating` + surface).

## C3 human beneficiaries (exact)

S01 returning astronaut; S02 isolated rescuer; S03 critically ill patient; S04 vehicle crew; S05 isolated station team; S06 critical patients on hospital life support; S07 vessel passengers and crew; S08 commuters and emergency personnel in tunnel; S09 residents of downstream communities; S10 residents on city drinking-water network; S11 residents evacuating threatened district; S12 occupants of surrounding buildings; S13 people relying on national grid for life-critical/essential services; S14 patients, households, essential-service users dependent on settlement; S15 aircraft passengers, mariners, emergency responders on navigation; S16 people consuming food in affected distribution; S17 people in affected cities exposed to outbreak risk; S18 passengers and flight crews in affected airspace; S19 refuge field personnel and nearby residents in environmental mandate; S20 marine field crews and coastal responders in operational mandate; S21 farm workers and regional communities in veterinary-surveillance mandate; S22 communities depending on ecosystem for water/flood/essential services; S23 disaster survivors and responders served by swarm; S24 emergency responders and communities on orbital network.

## Counts

24 families × 2 states = 48 instances; 24 YES + 24 NO.
