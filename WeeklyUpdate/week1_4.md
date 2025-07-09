# Project Update

Due Date: 07/16/2025
Done: No

# 🗓️ Weekly Update — [ADAS Project - ProPilot] - July 8, 2025

## ✅ Previous Weeks Progress

- [x]  Joystick Mapping([see details](https://github.com/adiautomotive/CARLA-Propilot/tree/devel_dara) )
- [x]  Display Image modification (BGRA → RGB)
- [x]  Tested my WaypointNAV code with the ProPiLOT LKA
    - [x]  Tested ACC accuracy
    - [x]  Tested LKA as WaypointNAV →
        - [x]  Speak to Adithya about “lane-check” function :
            - [x]  To activate handoff mode
                - [x]  ADAS_State.ACTIVE
                - [x]  lane_check(world)
                - [x]  upcoming_curvature < cornering_threshold
            - [x]  Update the lane-check based on Adithya reason
        - [x]  Design a scenario with degraded lane for LKA POC
- [x]  Designed the preliminary version of HUD tablet view
    
    ![HUD_VIEW](media/HUD_view.gif)
    

- [x]  Update the preliminary version of HUD tablet view
- [x]  Update GitHub repository
- [x]  Transfer code from GitHub repository to main computer in the LAB.

## 🔄 Ongoing Work

- [x]  Create New SettingMenu
- [x]  Combined the new setting
- [ ]  Modify map (Town 04) to add more the degraded lane for LKA
- [ ]  Add Traffic( Vehicle and Pedestrian)
- [ ]  Modify the dashboard display
    - [ ]  Include colors in the surrounding area
    - [ ]  Increase resolution and size relative to the Dashboard screen
- [ ]  Transfer code from GitHub repository to main computer in the LAB.
- [ ]  Create a human readable information enumerating the code functionalist

## Meeting Agenda

- [x]  Computer Setup( install map, setup correct version for Carla)
- [x]  Discuss progress on joystick mapping implementation
- [x]  Review image display modifications (BGRA to RGB)
- [x]  Discuss the ProPilot Features
    - [x]  ACC ( Adaptive  Cruise Control)
    - [x]  LKA (  Lane Keeping Assist )
    - [ ]  Other
        - [ ]  Traffic Jam Assist (TJA)
        - [ ]  Deceleration Assist
    - [ ]  Vehicle Mode/State transition
        - [x]  System OFF
        - [x]  Standby Mode
        - [x]  ICC
        - [ ]  ProPilot ON  → combine with ICC
        - [x]  Hands OFF Mode
    - [x]  Vehicle Scenarios  Test
        - [x]  Baseline Activation
        - [x]  Degraded Lane Markings (Fine-tuned and In progress)
        - [x]  Pedestrian or Object Cut-In  (In progress)
        - [x]  Stop-and-Go Exist  (Not yet)
        
    - [ ]  ProPILOT Assist (Base – SV & Rock Creek)
        - **Intelligent Cruise Control:** [Maintains safe following distance and adjusts speed in stop‑and‑go traffic—even comes to a complete stop and resumes with a tap](https://www.reddit.com/r/NissanRogue/comments/1benr3t/propilot_assist_what_yearsmodels_include_traffic/)
        - Lane Centering/Steering Assist: [Gently keeps the vehicle centered in its lane using camera and radar data](https://www.nissanofnewbraunfels.com/navigating-new-braunfels-nissan-propilot-assist/)
        - Automatic Braking : Applies brakes if it detects slower-moving traffic ahead
        - Stop & Hold :  Engages full stop in traffic and holds for up to ~30 seconds (resume via button)
        
        | Feature | ProPILOT | 1.1 | 2.1 (Tech Package) |
        | --- | --- | --- | --- |
        | Adaptive Cruise & Stop‑and‑Go | ✓ | ✓ | ✓ |
        | Lane Centering | ✓ | ✓ | ✓ |
        | Automatic Braking | ✓ | ✓ | ✓ |
        | Navi‑Link Speed & Curve Anticipation |  | ✓ | ✓ |
        | Speed Limit Recognition |  | ✓ | ✓ |
        | Hands‑Free Driving |  |  | ✓ |
        | Lane‑Change Assistance |  |  | ✓ |
        | Exit Awareness |  |  | ✓ |
        | Left‑Lane Return Prompt |  |  | ✓ |
        
- [x]  Present HUD tablet view preliminary design and updates
- [x]  Demo

## 📆 Next Week's Plan

- [ ]  List planned tasks
    - [ ]  Update the map to include interactive navigation message
    - [ ]  Update and fine-tuning the code for packaging into the window computer
        - [ ]  Add the modified map
        - [ ]  Add the modified Scenarios
        - [ ]  Improvise LKA logic
    - [ ]  Create another version of code with the feature separate activated with button  for comparative study
    - [ ]  
- [ ]  Mention deliverables
    - [ ]  A full function demo with all scenarios and features working properly