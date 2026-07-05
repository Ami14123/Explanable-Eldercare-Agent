# Planned Safe Navigation Agent

The Safe Navigation Agent is a planned prototype extension. It is not a current real navigation service.

## Intent Detection

Detect messages such as:

- "How do I get home?"
- "I am lost."
- "I need to go to the pharmacy."
- "Can you help me find the clinic?"

## Safety Risk Check

Before giving route-like support, check:

- dizziness, weakness, fall, confusion, panic, or getting lost
- whether the user is alone
- whether it is late, unsafe, or unfamiliar
- whether caregiver escalation is needed

## Missing Information

Ask for:

- current location
- destination
- whether the user can safely walk or wait
- whether they have permission to share location

## Mock Route Tool

Use `get_safe_route_tool` only as a mock structured tool. It does not call Google Maps or a real location service.

## Location Permission

Any future real integration must require explicit permission before using location data.

## Caregiver Escalation

Escalate to a caregiver or trusted person if:

- the user is lost or confused
- the user feels dizzy or unsafe walking
- the destination is urgent
- the user cannot provide reliable location details

## Privacy Limitations

The prototype should not store real addresses or live location. Public demos must use synthetic locations.
