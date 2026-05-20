# LLM-prompts-for-multibody-dynamics-models
LLM prompts using zero-shot, structured and RAG methods for generating Exudyn mechanical models.

Simple Pendulum
Zero-shot Prompt
Generate a complete and executable Python script using Exudyn to simulate the following simple pendulum model.

Target model:
- One point mass pendulum bob
- Pendulum length: 1.0 m
- Bob mass: 1.0 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle: 20 degrees from the vertical downward direction
- Motion in the x-y plane
- The bob is connected to ground at the origin using a distance constraint
- The pendulum angle must be computed as theta = atan2(x, -y)

Output:
- Export a CSV file named simple_pendulum_results.csv
- CSV columns: time, x, y, z, theta

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
Structured Prompt
Generate a complete and executable Python script using Exudyn to simulate the following simple pendulum model.

Target model:
- One point mass pendulum bob
- Pendulum length: 1.0 m
- Bob mass: 1.0 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle: 20 degrees from the vertical downward direction
- Motion in the x-y plane
- Ground point located at [0, 0, 0]
- Bob initial position:
  x = length * sin(theta0)
  y = -length * cos(theta0)
  z = 0

STRICT EXUDYN REQUIREMENTS:
- Use NodePointGround for the ground node
- Use NodePoint for the bob node
- Use MassPoint for the pendulum bob
- Use MarkerNodePosition for both ground and bob markers
- Use DistanceConstraint with markerNumbers=[groundMarker, bobMarker]
- Set the constraint distance to the pendulum length
- Apply gravity using Force with loadVector=[0, -mass*gravity, 0]
- Use SensorNode to record the bob position
- Use mbs.GetSensorStoredData to retrieve the stored position data
- Compute theta = atan2(x, -y)
- Use solverType=exu.DynamicSolverType.TrapezoidalIndex2

Output:
- Export a CSV file named simple_pendulum_results.csv
- CSV columns: time, x, y, z, theta

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
RAG Prompt
You are given a verified Exudyn reference example for a simple pendulum:

[  from __future__ import annotations

import math
from pathlib import Path

import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import (
    DistanceConstraint,
    Force,
    MarkerNodePosition,
    MassPoint,
    NodePoint,
    NodePointGround,
    ObjectGround,
    SensorNode,
    VMassPoint,
    VObjectGround,
)


def simulate_simple_pendulum(
    length: float = 1.0,
    mass: float = 1.0,
    gravity: float = 9.81,
    theta0_deg: float = 20.0,
    t_end: float = 10.0,
    step_size: float = 1e-3,
    output_path: str = "pendulum_solution.csv",
    show_viewer: bool = True,
) -> Path:
    """Run a simple pendulum simulation and export results to CSV.

    Args:
        length: Pendulum rod length in meters.
        mass: Bob mass in kilograms.
        gravity: Gravitational acceleration in m/s^2.
        theta0_deg: Initial angular displacement in degrees.
        t_end: Simulation end time in seconds.
        step_size: Integrator step size in seconds.
        output_path: Relative/absolute path for CSV output.
        show_viewer: If True, open the Exudyn simulation viewer window.

    Returns:
        Path to the generated CSV file.
    """

    theta0 = math.radians(theta0_deg)
    pin_radius = 0.03 * length
    bob_radius = 0.06 * length

    system_container = exu.SystemContainer()
    mbs = system_container.AddSystem()

    # Ground and bob nodes (3D, motion constrained by distance joint)
    n_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    n_bob = mbs.AddNode(
        NodePoint(
            referenceCoordinates=[
                length * math.sin(theta0),
                -length * math.cos(theta0),
                0.0,
            ],
            initialCoordinates=[0.0, 0.0, 0.0],
            initialVelocities=[0.0, 0.0, 0.0],
        )
    )

    # Visual pin support at origin and pendulum bob sphere.
    mbs.AddObject(
        ObjectGround(
            visualization=VObjectGround(
                graphicsData=[
                    graphics.Sphere(point=[0.0, 0.0, 0.0], radius=pin_radius, color=graphics.color.darkgrey)
                ]
            )
        )
    )
    mbs.AddObject(
        MassPoint(
            physicsMass=mass,
            nodeNumber=n_bob,
            visualization=VMassPoint(
                graphicsData=[
                    graphics.Sphere(point=[0.0, 0.0, 0.0], radius=bob_radius, color=graphics.color.dodgerblue)
                ]
            ),
        )
    )

    m_ground = mbs.AddMarker(MarkerNodePosition(nodeNumber=n_ground))
    m_bob = mbs.AddMarker(MarkerNodePosition(nodeNumber=n_bob))

    mbs.AddObject(DistanceConstraint(markerNumbers=[m_ground, m_bob], distance=length))
    # Gravity acts in -y direction; motion stays in the x-y plane.
    mbs.AddLoad(Force(markerNumber=m_bob, loadVector=[0.0, -mass * gravity, 0.0]))

    sensor_pos = mbs.AddSensor(
        SensorNode(
            nodeNumber=n_bob,
            outputVariableType=exu.OutputVariableType.Position,
            storeInternal=True,
        )
    )

    mbs.Assemble()

    simulation_settings = exu.SimulationSettings()
    simulation_settings.timeIntegration.startTime = 0.0
    simulation_settings.timeIntegration.endTime = t_end
    simulation_settings.timeIntegration.numberOfSteps = int(t_end / step_size)
    simulation_settings.solutionSettings.writeSolutionToFile = False
    simulation_settings.timeIntegration.simulateInRealtime = True
    simulation_settings.solutionSettings.sensorsWritePeriod = step_size
    simulation_settings.timeIntegration.verboseMode = 1

    # Viewer configuration
    system_container.visualizationSettings.general.autoFitScene = True
    system_container.visualizationSettings.nodes.show = False
    system_container.visualizationSettings.markers.show = False
    system_container.visualizationSettings.connectors.show = True
    system_container.visualizationSettings.loads.show = False

    renderer_started = False
    if show_viewer:
        # Exudyn >= 1.10 API (avoid deprecated global renderer functions)
        if hasattr(system_container, "renderer"):
            system_container.renderer.Start()
            renderer_started = True
            # Do initial idle processing; press SPACE in renderer to continue/advance.
            system_container.renderer.DoIdleTasks()
        else:
            # Fallback for older Exudyn versions
            exu.StartRenderer()
            renderer_started = True
            mbs.WaitForUserToContinue()

    try:
        mbs.SolveDynamic(
            simulation_settings,
            solverType=exu.DynamicSolverType.TrapezoidalIndex2,
        )
    finally:
        if renderer_started:
            if hasattr(system_container, "renderer"):
                system_container.renderer.Stop()
            else:
                exu.StopRenderer()

    data = mbs.GetSensorStoredData(sensor_pos)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("time,x,y,z,theta_rad\n")
        for row in data:
            t, x, y, z = row
            theta = math.atan2(x, -y)
            file.write(f"{t:.8f},{x:.8f},{y:.8f},{z:.8f},{theta:.8f}\n")

    return output


def main() -> None:
    result_file = simulate_simple_pendulum(
        length=1.0,
        mass=1.0,
        gravity=9.81,
        theta0_deg=20.0,
        t_end=10.0,
        step_size=1e-3,
        show_viewer=True,
    )
    print(f"Simulation complete. Results saved to: {result_file}")


if __name__ == "__main__":
    main()                  ]

Using the provided reference example, generate a complete and executable Python script for the same target model described below.

Target model:
- One point mass pendulum bob
- Pendulum length: 1.0 m
- Bob mass: 1.0 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle: 20 degrees from the vertical downward direction
- Motion in the x-y plane
- Ground point located at [0, 0, 0]
- Bob initial position:
  x = length * sin(theta0)
  y = -length * cos(theta0)
  z = 0
- The bob must be connected to ground using a DistanceConstraint
- The pendulum angle must be computed as theta = atan2(x, -y)

Use the reference example only as guidance for correct Exudyn API usage and program structure.

Output:
- Export a CSV file named simple_pendulum_results.csv
- CSV columns: time, x, y, z, theta

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
Double Pendulum
Zero-shot Prompt
Generate a complete and executable Python script using Exudyn to simulate the following double pendulum model.

Target model:
- Two point masses connected by two rigid distance constraints
- Ground point located at [0, 0, 0]
- Link 1 length: 1.0 m
- Link 2 length: 0.8 m
- Mass 1: 1.0 kg
- Mass 2: 0.8 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle of link 1: 30 degrees from the vertical downward direction
- Initial angle of link 2: 15 degrees from the vertical downward direction
- Motion in the x-y plane
- Mass 1 is connected to ground
- Mass 2 is connected to mass 1
- Compute:
  theta1 = atan2(x1, -y1)
  theta2 = atan2(x2 - x1, -(y2 - y1))

Output:
- Export a CSV file named double_pendulum_results.csv
- CSV columns: time, x1, y1, z1, theta1, x2, y2, z2, theta2

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
Structured Prompt
Generate a complete and executable Python script using Exudyn to simulate the following double pendulum model.

Target model:
- Two point masses connected by two rigid distance constraints
- Ground point located at [0, 0, 0]
- Link 1 length: 1.0 m
- Link 2 length: 0.8 m
- Mass 1: 1.0 kg
- Mass 2: 0.8 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle of link 1: 30 degrees from the vertical downward direction
- Initial angle of link 2: 15 degrees from the vertical downward direction
- Motion in the x-y plane

Initial coordinates:
- x1 = length1 * sin(theta1)
- y1 = -length1 * cos(theta1)
- z1 = 0
- x2 = x1 + length2 * sin(theta2)
- y2 = y1 - length2 * cos(theta2)
- z2 = 0

STRICT EXUDYN REQUIREMENTS:
- Use NodePointGround for the ground node
- Use two NodePoint objects for the two masses
- Use two MassPoint objects
- Use MarkerNodePosition for ground, mass 1, and mass 2
- Use one DistanceConstraint between ground and mass 1
- Use one DistanceConstraint between mass 1 and mass 2
- Apply gravity to mass 1 using Force with loadVector=[0, -mass1*gravity, 0]
- Apply gravity to mass 2 using Force with loadVector=[0, -mass2*gravity, 0]
- Use SensorNode to record the position of mass 1
- Use SensorNode to record the position of mass 2
- Use mbs.GetSensorStoredData to retrieve both position histories
- Compute:
  theta1 = atan2(x1, -y1)
  theta2 = atan2(x2 - x1, -(y2 - y1))
- Use solverType=exu.DynamicSolverType.TrapezoidalIndex2

Output:
- Export a CSV file named double_pendulum_results.csv
- CSV columns: time, x1, y1, z1, theta1, x2, y2, z2, theta2

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
RAG Prompt
You are given a verified Exudyn reference example for a double pendulum or closely related multibody system:

[ from __future__ import annotations

import math
from pathlib import Path

import exudyn as exu
import exudyn.graphics as graphics
from exudyn.itemInterface import (
    DistanceConstraint,
    Force,
    MarkerNodePosition,
    MassPoint,
    NodePoint,
    NodePointGround,
    ObjectGround,
    SensorNode,
    VMassPoint,
    VObjectGround,
)


def simulate_double_pendulum(
    length1: float = 1.0,
    length2: float = 1.0,
    mass1: float = 1.0,
    mass2: float = 1.0,
    gravity: float = 9.81,
    theta1_deg: float = 30.0,
    theta2_deg: float = 15.0,
    t_end: float = 20.0,
    step_size: float = 1e-3,
    output_path: str = "double_pendulum_solution.csv",
    show_viewer: bool = True,
) -> Path:
    """Run a double pendulum simulation and export results to CSV.

    Angles are measured from vertical downward direction in the x-y plane.
    """

    theta1 = math.radians(theta1_deg)
    theta2 = math.radians(theta2_deg)

    pin_radius = 0.03 * max(length1, length2)
    bob1_radius = 0.06 * length1
    bob2_radius = 0.06 * length2

    x1 = length1 * math.sin(theta1)
    y1 = -length1 * math.cos(theta1)
    x2 = x1 + length2 * math.sin(theta2)
    y2 = y1 - length2 * math.cos(theta2)

    system_container = exu.SystemContainer()
    mbs = system_container.AddSystem()

    n_ground = mbs.AddNode(NodePointGround(referenceCoordinates=[0.0, 0.0, 0.0]))
    n_bob1 = mbs.AddNode(
        NodePoint(
            referenceCoordinates=[x1, y1, 0.0],
            initialCoordinates=[0.0, 0.0, 0.0],
            initialVelocities=[0.0, 0.0, 0.0],
        )
    )
    n_bob2 = mbs.AddNode(
        NodePoint(
            referenceCoordinates=[x2, y2, 0.0],
            initialCoordinates=[0.0, 0.0, 0.0],
            initialVelocities=[0.0, 0.0, 0.0],
        )
    )

    mbs.AddObject(
        ObjectGround(
            visualization=VObjectGround(
                graphicsData=[
                    graphics.Sphere(
                        point=[0.0, 0.0, 0.0],
                        radius=pin_radius,
                        color=graphics.color.darkgrey,
                    )
                ]
            )
        )
    )
    mbs.AddObject(
        MassPoint(
            physicsMass=mass1,
            nodeNumber=n_bob1,
            visualization=VMassPoint(
                graphicsData=[
                    graphics.Sphere(
                        point=[0.0, 0.0, 0.0],
                        radius=bob1_radius,
                        color=graphics.color.dodgerblue,
                    )
                ]
            ),
        )
    )
    mbs.AddObject(
        MassPoint(
            physicsMass=mass2,
            nodeNumber=n_bob2,
            visualization=VMassPoint(
                graphicsData=[
                    graphics.Sphere(
                        point=[0.0, 0.0, 0.0],
                        radius=bob2_radius,
                        color=graphics.color.red,
                    )
                ]
            ),
        )
    )

    m_ground = mbs.AddMarker(MarkerNodePosition(nodeNumber=n_ground))
    m_bob1 = mbs.AddMarker(MarkerNodePosition(nodeNumber=n_bob1))
    m_bob2 = mbs.AddMarker(MarkerNodePosition(nodeNumber=n_bob2))

    mbs.AddObject(DistanceConstraint(markerNumbers=[m_ground, m_bob1], distance=length1))
    mbs.AddObject(DistanceConstraint(markerNumbers=[m_bob1, m_bob2], distance=length2))

    mbs.AddLoad(Force(markerNumber=m_bob1, loadVector=[0.0, -mass1 * gravity, 0.0]))
    mbs.AddLoad(Force(markerNumber=m_bob2, loadVector=[0.0, -mass2 * gravity, 0.0]))

    sensor_bob1 = mbs.AddSensor(
        SensorNode(
            nodeNumber=n_bob1,
            outputVariableType=exu.OutputVariableType.Position,
            storeInternal=True,
        )
    )
    sensor_bob2 = mbs.AddSensor(
        SensorNode(
            nodeNumber=n_bob2,
            outputVariableType=exu.OutputVariableType.Position,
            storeInternal=True,
        )
    )

    mbs.Assemble()

    simulation_settings = exu.SimulationSettings()
    simulation_settings.timeIntegration.startTime = 0.0
    simulation_settings.timeIntegration.endTime = t_end
    simulation_settings.timeIntegration.numberOfSteps = int(t_end / step_size)
    simulation_settings.solutionSettings.writeSolutionToFile = False
    simulation_settings.timeIntegration.simulateInRealtime = True
    simulation_settings.solutionSettings.sensorsWritePeriod = step_size
    simulation_settings.timeIntegration.verboseMode = 1

    system_container.visualizationSettings.general.autoFitScene = True
    system_container.visualizationSettings.nodes.show = False
    system_container.visualizationSettings.markers.show = False
    system_container.visualizationSettings.connectors.show = True
    system_container.visualizationSettings.loads.show = False

    renderer_started = False
    if show_viewer:
        if hasattr(system_container, "renderer"):
            system_container.renderer.Start()
            renderer_started = True
            system_container.renderer.DoIdleTasks()
        else:
            exu.StartRenderer()
            renderer_started = True
            mbs.WaitForUserToContinue()

    try:
        mbs.SolveDynamic(
            simulation_settings,
            solverType=exu.DynamicSolverType.TrapezoidalIndex2,
        )
    finally:
        if renderer_started:
            if hasattr(system_container, "renderer"):
                system_container.renderer.Stop()
            else:
                exu.StopRenderer()

    data1 = mbs.GetSensorStoredData(sensor_bob1)
    data2 = mbs.GetSensorStoredData(sensor_bob2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("time,x1,y1,z1,theta1_rad,x2,y2,z2,theta2_rad\n")
        for row1, row2 in zip(data1, data2):
            t, x1, y1, z1 = row1
            _, x2, y2, z2 = row2
            theta1 = math.atan2(x1, -y1)
            theta2 = math.atan2(x2 - x1, -(y2 - y1))
            file.write(
                f"{t:.8f},{x1:.8f},{y1:.8f},{z1:.8f},{theta1:.8f},"
                f"{x2:.8f},{y2:.8f},{z2:.8f},{theta2:.8f}\n"
            )

    return output


def main() -> None:
    result_file = simulate_double_pendulum(
        length1=1.0,
        length2=0.8,
        mass1=1.0,
        mass2=0.8,
        gravity=9.81,
        theta1_deg=30.0,
        theta2_deg=15.0,
        t_end=10.0,
        step_size=1e-3,
        show_viewer=True,
    )
    print(f"Simulation complete. Results saved to: {result_file}")


if __name__ == "__main__":
    main()     ]

Using the provided reference example, generate a complete and executable Python script for the same target model described below.

Target model:
- Two point masses connected by two rigid distance constraints
- Ground point located at [0, 0, 0]
- Link 1 length: 1.0 m
- Link 2 length: 0.8 m
- Mass 1: 1.0 kg
- Mass 2: 0.8 kg
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial angle of link 1: 30 degrees from the vertical downward direction
- Initial angle of link 2: 15 degrees from the vertical downward direction
- Motion in the x-y plane

Initial coordinates:
- x1 = length1 * sin(theta1)
- y1 = -length1 * cos(theta1)
- z1 = 0
- x2 = x1 + length2 * sin(theta2)
- y2 = y1 - length2 * cos(theta2)
- z2 = 0

Use the reference example only as guidance for correct Exudyn API usage and program structure.

Output:
- Export a CSV file named double_pendulum_results.csv
- CSV columns: time, x1, y1, z1, theta1, x2, y2, z2, theta2

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
Mass-Spring-Damper System
Zero-shot Prompt
Generate a complete and executable Python script using Exudyn to simulate the following mass-spring-damper system.

Target model:
- One point mass connected to ground using a spring-damper element
- Motion is 1D vertical motion in the y-direction
- Ground point located at [0, 0, 0]
- Mass reference position: [0, -0.5, 0]
- Mass: 1.0 kg
- Spring stiffness: 1000 N/m
- Damping coefficient: 5 Ns/m
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial displacement from reference position: -0.05 m in the y-direction
- Initial velocity: 0 m/s
- Record vertical displacement y and vertical velocity vy

Output:
- Export a CSV file named mass_spring_damper_results.csv
- CSV columns: time, y, vy

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
Structured Prompt
Generate a complete and executable Python script using Exudyn to simulate the following mass-spring-damper system.

Target model:
- One point mass connected to ground using a spring-damper element
- Motion is 1D vertical motion in the y-direction
- Ground point located at [0, 0, 0]
- Mass reference position: [0, -0.5, 0]
- Mass: 1.0 kg
- Spring stiffness: 1000 N/m
- Damping coefficient: 5 Ns/m
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial displacement from reference position: -0.05 m in the y-direction
- Initial velocity: 0 m/s

STRICT EXUDYN REQUIREMENTS:
- Use NodePointGround for the ground node
- Use NodePoint for the moving mass node
- Use MassPoint for the moving mass
- Use a spring-damper connector between ground and the mass
- Use coordinate-based markers for the vertical y-coordinate if using CoordinateSpringDamper
- The spring-damper must act in the vertical y-direction
- Apply gravity in the vertical y-direction
- Use sensors or node output data to record vertical position y and vertical velocity vy
- Use solverType=exu.DynamicSolverType.TrapezoidalIndex2

Output:
- Export a CSV file named mass_spring_damper_results.csv
- CSV columns: time, y, vy

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
RAG Prompt
You are given a verified Exudyn reference example for a spring-damper oscillator:

[  import exudyn as exu
from exudyn.itemInterface import *
from exudyn.utilities import * #includes itemInterface and rigidBodyUtilities
import exudyn.graphics as graphics #only import if it does not conflict

## set up system 'mbs'
SC = exu.SystemContainer()
mbs = SC.AddSystem()

## define overall parameters for linear spring-damper
L=0.5
mass = 1.6
k = 4000
omega0 = 50 # sqrt(4000/1.6)
dRel = 0.05
d = dRel * 2 * 80 #80=sqrt(1.6*4000)
u0=-0.08
v0=1
f = 80
#static result = f/k = 0.01; 
x0 = f/k

## create ground object
objectGround = mbs.CreateGround(referencePosition = [0,0,0])

## create mass point
massPoint = mbs.CreateMassPoint(referencePosition=[L,0,0],
                    initialDisplacement=[u0,0,0],
                    initialVelocity=[v0,0,0],
                    physicsMass=mass)

## create spring damper  between objectGround and massPoint
mbs.CreateCartesianSpringDamper(bodyNumbers=[objectGround, massPoint],
                                stiffness = [k,k,k], 
                                damping   = [d,0,0], 
                                offset    = [L,0,0])

## create force vector [f,0,0]
mbs.CreateForce(bodyNumber=massPoint, loadVector= [f,0,0])

## assemble and solve system
mbs.Assemble()

simulationSettings = exu.SimulationSettings()

tEnd = 1
steps = 1000000
simulationSettings.timeIntegration.numberOfSteps = steps
simulationSettings.timeIntegration.endTime = tEnd
simulationSettings.timeIntegration.newton.useModifiedNewton = True
simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 1 #SHOULD work with 0.9 as well

## solve and read displacement at end time
mbs.SolveDynamic(simulationSettings)
uCSD = mbs.GetObjectOutputBody(massPoint, exu.OutputVariableType.Displacement)[0]

## compute exact (analytical) solution:
import numpy as np
import matplotlib.pyplot as plt

# omega0 = np.sqrt(k/mass)     
# dRel = d/(2*np.sqrt(k*mass)) 

omega = omega0*np.sqrt(1-dRel**2)
C1 = u0-x0 #static solution needs to be considered!
C2 = (v0+omega0*dRel*C1) / omega #C1 used instead of classical solution with u0, because x0 != 0 !!!

refSol = np.zeros((steps+1,2))
for i in range(0,steps+1):
    t = tEnd*i/steps
    refSol[i,0] = t
    refSol[i,1] = np.exp(-omega0*dRel*t)*(C1*np.cos(omega*t) + C2*np.sin(omega*t))+x0

print('refSol=',refSol[steps,1])
print('error exact-numerical=',refSol[steps,1] - uCSD)

## compare Exudyn with analytical solution:
mbs.PlotSensor(['coordinatesSolution.txt', refSol],
                components=[0,0], yLabel='displacement',
                labels=['Exudyn','analytical'])    ]

Using the provided reference example, generate a complete and executable Python script for the same target model described below.

Target model:
- One point mass connected to ground using a spring-damper element
- Motion is 1D vertical motion in the y-direction
- Ground point located at [0, 0, 0]
- Mass reference position: [0, -0.5, 0]
- Mass: 1.0 kg
- Spring stiffness: 1000 N/m
- Damping coefficient: 5 Ns/m
- Gravity: 9.81 m/s^2 acting in the negative y-direction
- Initial displacement from reference position: -0.05 m in the y-direction
- Initial velocity: 0 m/s
- Record vertical displacement y and vertical velocity vy

Important:
- The retrieved example may use x-direction motion, plotting, renderer settings, or additional force terms.
- Do not copy those parts unless they are needed.
- Adapt the model so that the final system uses vertical y-direction motion and exports only the requested CSV file.
- Use the reference example only as guidance for correct Exudyn API usage and spring-damper structure.

Output:
- Export a CSV file named mass_spring_damper_results.csv
- CSV columns: time, y, vy

Use the following fixed simulation parameters:
- Simulation time: 10 s
- Step size: 0.001 s
- Solver: TrapezoidalIndex2
- Do not open the Exudyn renderer/viewer
- Do not include plots
- Export only the required CSV file
- The script must be fully executable from a clean Python file
- Include all required imports
- Do not use pseudocode
