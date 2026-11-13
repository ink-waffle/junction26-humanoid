# Aaltonomous Humanoid (Aalto University): текст челленджа

Скопировано со страницы челленджа на платформе Junction 13.11.2026.

Aaltonomous Humanoid
Aaltonomous Humanoid
by Aalto University
Our humanoid robot can do backflips but can't sort a few blocks into boxes. Give it a brain.
Introduction
Humanoid robot hardware has improved enormously in recent years. Robots can now do impressive things like backflips, dancing and kung fu. They are also becoming surprisingly affordable: the Unitree G1 costs around €50,000, and it won't be long before humanoids are within reach of ordinary people.
But there's a catch, known in robotics as Moravec's paradox. Hard-coded backflips are relatively easy for a robot, while something as simple as picking up a small block and putting it in the right box is much harder, even though a small child can do it without thinking. Our G1, Väiski, is physically very capable, but it's still pretty dumb: it has very little autonomy or onboard intelligence.
During this hackathon, your job is to give it a brain.
The task
Make the G1 sort objects into boxes on its own, without a human controlling it.
Väiski stands behind a counter with its legs and waist held stiff, working with two 7-DoF arms and two Dex3 three-fingered hands. In front of it, 3 to 5 printed objects lie at random positions and angles: cubes, cylinders and triangular prisms, about 4.5 cm in size, in red, yellow and blue.
Behind the objects are three identical orange boxes (21 cm wide) at fixed positions: left, centre and right, in a row, 5 cm apart from each other and 40 cm from the robot's torso. The boxes have no markings; they are told apart only by their position. The robot has to put every object into the right box within 2 minutes.
Fixed sorting: red objects go into the left box, yellow into the centre box, blue into the right box.
Sort by instruction: a text instruction defines which box each object goes into, e.g. "Sort the objects: red into right, blue into left, yellow into center." Expect phrasings you haven't seen before.
Stretch goal: put each object through the matching hole in a lid on one box.
Your policy sees the head camera (colour + depth), both wrist cameras, the joint states and the instruction, and sends back target positions for all arm and finger joints, 30 times a second.
How it works
Simulation first: build and test your solution in a ready-made Isaac Sim simulation of the robot and our exact scene, with plenty of GPU power from Verda.
Real-world data during the event: you don't need any data to get started. During the hackathon, the Aalto University team will publish datasets recorded with the real robot at the venue on Hugging Face, with links shared in the Junction Discord. You can use them for training and for tuning your solution to the real cameras and conditions.
Submit a Docker image: your solution runs as a small policy server in a container. Our runner sends its commands to the simulator or the robot through our safety filter.
Finals: we evaluate submissions in simulation, on a version of the scene you can't train on, and pick the finalists. They run on the real robot on Sunday. The best real-robot score wins.
Possible directions
Any approach is welcome: computer vision and classic programming, a VLM/LLM planner calling your own skills, imitation learning or VLA fine-tuning (LeRobot, π0, GR00T), reinforcement learning in Isaac Lab, making the simulation match reality, or something no one has tried yet. Combinations will probably work best.
What we'd love to see
Solutions that still work when the real world doesn't match the simulation exactly.
General ideas that could work beyond a single task, rather than ones tuned to one exact setup.
Clever ways to do a lot with very little real-world data.
What we'll bring
A real humanoid robot: the Unitree G1 with Dex3 three-fingered hands, live at the venue.
A simulator: a ready-to-use Docker image with Isaac Sim, Isaac Lab and our challenge scene.
GPU power: compute from Verda.
Real robot data: datasets recorded with the real robot, published on Hugging Face during the hackathon (links in the Junction Discord).
Starter code: the robot interface, the safety filter, a classical baseline, policy-server and Dockerfile templates, and an evaluator that scores your solution in simulation.
Expert mentors: researchers who work with the robot every day, plus around 30 Aalto Robotics Club volunteers.
Starter repository
https://github.com/zupermiez/aaltonomous-humanoid
The step-by-step guide is in aaltonomous/cloud/QUICKSTART.md; rules, interface and submission details are in aaltonomous/docs/participants/.
Getting started in simulation
GPU: use an RTX-class GPU (RTX Pro 6000, L40S, A6000, RTX 4090/5090, ≥ 24 GB). The simulator renders the cameras with ray tracing, so it will not render on A100 or H100 (they are fine for training). Use on-demand instances, not spot, and keep your work on a persistent volume.
Start the simulator:
docker load -i aalto-sim-v1.tar.zst      # download location shared at kickoff
docker run -d --name aalto --gpus all --network host --ipc host \
    -v aalto-kitcache:/root/miniforge3/envs/isaac/lib/python3.12/site-packages/isaacsim/kit/cache \
    -v /data:/data aalto-sim:v1 sleep infinity
docker exec -it aalto bash
aalto sim        # start the simulator (first start ~5-10 min while shaders compile)
aalto check      # scene checks: 8 PASS / 0 FAIL
aalto baseline   # run a reference policy and print a score table
No Docker? Clone the repo and run bash aaltonomous/cloud/setup_instance.sh (about 30–40 minutes, no sudo needed).
Then: write your policy as a server from the template in aaltonomous/deploy/, score it in the simulator with aalto eval --policy ws://localhost:8000, and package it with aaltonomous/docker/submission.Dockerfile. To run on the real robot, a solution must pass our safety check in simulation: no crashes or safety stops, no commands over the joint and speed limits, and at least 27 commands per second.
Mentors
Erkka Lehtonen and Viktor Solianoi, on site throughout the event. Questions and announcements: the Junction Discord.
Safety
Teams never control the robot directly. Our runner executes your Docker image through our safety filter (joint and speed limits, workspace limits, a watchdog) while our team stands at the emergency stop. Code has to pass the safety checks in simulation before it runs on the real robot.
Aalto University
Aalto University is where people who think differently bring out the best in each other — where imagination meets reality and challenges once thought impossible are solved. Born to unite science, arts, technology, and business, Aalto exists to create a better world by bringing diverse minds together — from quantum engineers to fashion designers. We accelerate curiosity and responsibility, reimagining how a university can operate. Driven by radical creativity and an entrepreneurial mindset, we turn theories into solutions and prototypes into breakthroughs. Our differences spark action and collaboration. Together, we build beyond the status quo and always find a way forward. The AR4U project at Aalto University explores how voice-based artificial intelligence can help older adults stay active, connected, and independent at home. The project develops and tests an AI companion and calling service that provides daily support, conversation, and well-being check-ins—making technology more personal, accessible, and engaging for everyone. AR4U is an Research-to-Business (R2B) project at Aalto University, funded by Business Finland, aiming to transform academic research into commercial solutions.
Website: https://aalto.fi/
Real-world result
(40%)
How well it works on the real robot
Generality
(25%)
Holds up when things change
Originality
(20%)
Freshness of idea and approach
Presentation
(15%)
Clarity of what you built and why it matters
Winning Team
2,000€ + a visit to the Aalto robotics lab
Top 5
🤖 **Top ~5 teams** (chosen by performance in simulation) get to run their solution **live on the real Unitree G1**.