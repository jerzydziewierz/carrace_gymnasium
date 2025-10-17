"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math

import Box2D
import numpy as np

from gymnasium.error import DependencyNotInstalled

print(f'loading Car Dynamics module gymnasium.envs.box2d.car_dynamics')
try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e



SIZE = 0.02
ENGINE_POWER = 0.75*100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
BRAKE_FORCE = 30  # radians per second
from car_constants import *


print(f'FRICTION_LIMIT set to {FRICTION_LIMIT}')
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
#WHEELPOS = [(-55, +80), (+55, +80), (-85, -82), (+85, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)



class Car:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world: Box2D.b2World = world
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY4]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * SIZE, y * front_k * SIZE)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation
        """
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side
        """
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for this_wheel in self.wheels:            
            # Steer each wheel
            dir = np.sign(this_wheel.steer - this_wheel.joint.angle)
            val = abs(this_wheel.steer - this_wheel.joint.angle)
            this_wheel.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            is_grass = False
            is_ice = False
            is_road = False

            # friction_limit = FRICTION_LIMIT * 0.1  # Grass friction if no tile
            friction_limit = FRICTION_LIMIT # start from very high friction, can be decreased by tiles below

            print(f'Wheel at position {this_wheel.position} touching {len(this_wheel.tiles)} tiles')
            for tile in this_wheel.tiles:                
                if tile.gtype == TILETYPE_ROAD:  # road
                    is_road = True
                if tile.gtype == TILETYPE_GRASS:  # grass
                    is_grass = True
                if tile.gtype == TILETYPE_ICE:  # ice
                    is_ice = True

            if is_road:
                is_grass = False # road has priority over grass         
            
            if is_ice:
                is_grass = False
                is_road = False # ice has priority over road and grass
            
            if not (is_road or is_grass or is_ice):
                friction_limit = FRICTION_LIMIT * FRICTION_BASE_GRASS  # default to grass
            elif is_road:
                friction_limit = FRICTION_LIMIT * FRICTION_BASE_ROAD
            elif is_grass:
                friction_limit = FRICTION_LIMIT * FRICTION_BASE_GRASS
            elif is_ice:
                friction_limit = FRICTION_LIMIT * FRICTION_BASE_ICE

            # Force
            forw = this_wheel.GetWorldVector((0, 1))
            side = this_wheel.GetWorldVector((1, 0))
            v = this_wheel.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            this_wheel.omega += (
                dt
                * ENGINE_POWER
                * this_wheel.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(this_wheel.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * this_wheel.gas

            if this_wheel.brake >= 0.9:
                this_wheel.omega = 0
            elif this_wheel.brake > 0:

                dir = -np.sign(this_wheel.omega)
                val = BRAKE_FORCE * this_wheel.brake
                if abs(val) > abs(this_wheel.omega):
                    val = abs(this_wheel.omega)  # low speed => same as = 0
                this_wheel.omega += dir * val
            this_wheel.phase += this_wheel.omega * dt

            vr = this_wheel.omega * this_wheel.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    this_wheel.skid_particle
                    and this_wheel.skid_particle.grass == is_grass
                    and len(this_wheel.skid_particle.poly) < SKID_MARK_MEMORY_LENGTH
                ):
                    this_wheel.skid_particle.poly.append((this_wheel.position[0], this_wheel.position[1]))
                elif this_wheel.skid_start is None:
                    this_wheel.skid_start = this_wheel.position
                else:
                    this_wheel.skid_particle = self._create_particle(
                        this_wheel.skid_start, this_wheel.position, is_grass, is_ice
                    )
                    this_wheel.skid_start = None
            else:
                this_wheel.skid_start = None
                this_wheel.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            this_wheel.omega -= dt * f_force * this_wheel.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            this_wheel.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=int(WHEEL_W/2), closed=False
                )


        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                color = [int(c * 255) for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                    (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                ]
                white_poly = [trans * v for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

    def _create_particle(self, point1, point2, is_grass, is_ice=False):
        class Particle:
            pass

        p = Particle()
        p.color = WHEEL_COLOR if not is_grass else MUD_COLOR
        if is_ice:
            p.color = ICE_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = is_grass or is_ice
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
