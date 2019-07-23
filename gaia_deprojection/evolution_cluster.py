import numpy
from amuse.couple import bridge
from amuse.units import units, constants, quantities, nbody_system
from amuse.ic.brokenimf import new_broken_power_law_mass_distribution
from amuse.ic.plummer import new_plummer_sphere
from amuse.community.huayno.interface import Huayno
from amuse.community.seba.interface import SeBa
from amuse.ext.composition_methods import *
from amuse.ext.rotating_bridge import Rotating_Bridge
from amuse.community.galaxia.interface import BarAndSpirals3D
import matplotlib
import pandas as pd
import argparse
import logging
from matplotlib import pyplot
import h5py
import pylab as plt
from pandas import HDFStore, DataFrame
import os
import numpy as np
from timeit import default_timer
import scipy.stats as st
import dask

# create (or open) an hdf5 file and opens in append mode
hdf = HDFStore('storage.h5')


def create_cluster_with_IMF(N=100, radius=3 | units.parsec):
    '''
    Creation of a cluster following a Kroupa IMF
    '''
    masses = new_broken_power_law_mass_distribution(N,
                                                    mass_boundaries=[0.08, 0.5, 100] | units.MSun,
                                                    alphas=[-1.3, -2.3])
    convert_nbody = nbody_system.nbody_to_si(masses.sum(), radius)
    cluster = new_plummer_sphere(N, convert_nbody)
    cluster.mass = masses
    cluster.move_to_center()
    cluster.scale_to_standard(convert_nbody)
    return cluster


class RealisticEvolutionCluster(object):
    """
    This class makes the integration of a star cluster 
    in an analytical potential using galaxia.
    The integration methods that can be used in the Rotating Bridge are:
    LEAPFROG
    SPLIT_4TH_S_M6
    SPLIT_4TH_S_M5
    SPLIT_4TH_S_M4
    SPLIT_6TH_SS_M11
    SPLIT_6TH_SS_M13
    SPLIT_8TH_SS_M21
    SPLIT_10TH_SS_M35
    where the ordinal number stands for the order of the integrator (i.e, 4th is fourth order);
    the S for symplectic; and M corresponds to the number of times the force in computed
    (i.e., M6 means that the force is computed 6 times).
    """

    def __init__(self, simulation_time=100 | units.Myr,
                 dt_bridge=1 | units.Myr,
                 method_for_rotating_bridge=LEAPFROG,
                 initial_phase_main_spiral_arms=0,
                 pattern_speed_main_spiral_arms=20 | (units.kms / units.kpc),
                 amplitude_main_spiral_arms=1100 | (units.kms ** 2 / units.kpc),
                 number_of_main_spiral_arms=2,
                 tangent_pitch_angle_main_spiral_arms=0.227194425,
                 initial_phase_secondary_spiral_arms=200 * (numpy.pi / 180),
                 pattern_speed_secondary_spiral_arms=15 | (units.kms / units.kpc),
                 amplitude_secondary_spiral_arms=880 | (units.kms ** 2 / units.kpc),
                 tangent_pitch_angle_secondary_spiral_arms=numpy.tan((-14 * numpy.pi) / 180.),
                 number_of_secondary_spiral_arms=2,
                 separation_locus_spiral_arms=3.12 | units.kpc,
                 initial_phase_bar=0,
                 pattern_speed_bar=40 | (units.kms / units.kpc),
                 mass_bar=1.2e10 | units.MSun,
                 semimajor_axis_bar=3.12 | units.kpc,
                 axis_ratio_bar=0.37):

        # Simulation parameters
        self.t_end = simulation_time
        self.time = 0 | units.Myr
        self.dt_bridge = dt_bridge
        self.method = method_for_rotating_bridge

        # galaxy parameters
        self.omega_system = 0 | (units.kms / units.kpc)
        self.initial_phase_system = 0
        self.bar_phase = initial_phase_bar
        self.omega_bar = pattern_speed_bar
        self.mass_bar = mass_bar
        self.aaxis_bar = semimajor_axis_bar
        self.axis_ratio_bar = axis_ratio_bar

        self.spiral_phase_main_sp = initial_phase_main_spiral_arms
        self.omega_spiral_main_sp = pattern_speed_main_spiral_arms
        self.amplitude_main_sp = amplitude_main_spiral_arms
        self.number_main_sp = number_of_main_spiral_arms
        self.tangent_pitch_angle_main_sp = tangent_pitch_angle_main_spiral_arms

        self.spiral_phase_second_sp = initial_phase_secondary_spiral_arms
        self.omega_spiral_second_sp = pattern_speed_secondary_spiral_arms
        self.amplitude_second_sp = amplitude_secondary_spiral_arms
        self.number_second_sp = number_of_secondary_spiral_arms
        self.tangent_pitch_angle_second_sp = tangent_pitch_angle_secondary_spiral_arms
        self.separation_sp = separation_locus_spiral_arms

        return

    def softening(self, particles):
        '''
        optimum softening lenght.
        '''
        N = len(particles.mass)
        U = particles.potential_energy()
        Rvir = 0.5 * constants.G * particles.mass.sum() ** 2 / abs(U)
        epsilon = 4 * Rvir / N
        return epsilon

    def galactic_model(self):
        '''
         Model of the Galaxy.
         In this example, the Galaxy has two-dimensional bar and spiral arms.
         The spiral arms are described by the Composite model (2+2)
         The bar does not grow adiabatically
         The axisymmetric component has its defaul values from Allen & Santillan (1990).
         '''
        galaxy = BarAndSpirals3D()
        galaxy.kinetic_energy = quantities.zero
        galaxy.potential_energy = quantities.zero
        galaxy.parameters.spiral_contribution = True
        galaxy.parameters.spiral_model = 2
        galaxy.parameters.omega_spiral = self.omega_spiral_main_sp
        galaxy.parameters.spiral_phase = self.spiral_phase_main_sp
        galaxy.parameters.amplitude = self.amplitude_main_sp
        galaxy.parameters.m = self.number_main_sp
        galaxy.parameters.tan_pitch_angle = self.tangent_pitch_angle_main_sp
        galaxy.parameters.phi21_spiral = self.spiral_phase_second_sp
        galaxy.parameters.omega_spiral2 = self.omega_spiral_second_sp
        galaxy.parameters.amplitude2 = self.amplitude_second_sp
        galaxy.parameters.m2 = self.number_second_sp
        galaxy.parameters.tan_pitch_angle2 = self.tangent_pitch_angle_second_sp
        galaxy.parameters.rsp = self.separation_sp
        galaxy.parameters.bar_contribution = True
        galaxy.parameters.bar_phase = self.bar_phase
        galaxy.parameters.omega_bar = self.omega_bar
        galaxy.parameters.mass_bar = self.mass_bar
        galaxy.parameters.aaxis_bar = self.aaxis_bar
        galaxy.parameters.axis_ratio_bar = self.axis_ratio_bar
        galaxy.commit_parameters()
        self.omega_system = galaxy.parameters.omega_system
        self.initial_phase_sytem = galaxy.parameters.initial_phase

        return galaxy

    def circular_velocity(self):
        MW = self.galactic_model()
        r = numpy.arange(15)
        vc = MW.get_velcirc(r | units.kpc, 0 | units.kpc, 0 | units.kpc)
        pyplot.plot(r, vc.value_in(units.kms))
        pyplot.show()

    def creation_cluster_in_rotating_frame(self, particles):
        "forming a cluster in a rotating frame"

        no_inertial_system = particles.copy()
        angle = self.initial_phase_system + self.omega_system * self.time
        C1 = particles.vx + self.omega_system * particles.y
        C2 = particles.vy - self.omega_system * particles.x
        no_inertial_system.x = particles.x * numpy.cos(angle) + particles.y * numpy.sin(angle)
        no_inertial_system.y = -particles.x * numpy.sin(angle) + particles.y * numpy.cos(angle)
        no_inertial_system.z = particles.z
        no_inertial_system.vx = C1 * numpy.cos(angle) + C2 * numpy.sin(angle)
        no_inertial_system.vy = C2 * numpy.cos(angle) - C1 * numpy.sin(angle)
        no_inertial_system.vz = particles.vz
        return no_inertial_system

    def from_noinertial_to_cluster_in_inertial_frame(self, part_noin, part_in):
        'makes transformation to the inertial frame'

        angle = self.initial_phase_system + self.omega_system * self.time
        C1 = part_noin.vx - part_noin.y * self.omega_system
        C2 = part_noin.vy + part_noin.x * self.omega_system
        part_in.age = part_noin.age
        part_in.mass = part_noin.mass
        part_in.radius = part_noin.radius
        part_in.luminosity = part_noin.luminosity
        part_in.temperature = part_noin.temperature
        part_in.stellar_type = part_noin.stellar_type
        part_in.x = part_noin.x * numpy.cos(angle) - part_noin.y * numpy.sin(angle)
        part_in.y = part_noin.x * numpy.sin(angle) + part_noin.y * numpy.cos(angle)
        part_in.z = part_noin.z
        part_in.vx = C1 * numpy.cos(angle) - C2 * numpy.sin(angle)
        part_in.vy = C1 * numpy.sin(angle) + C2 * numpy.cos(angle)
        part_in.vz = part_noin.vz
        return

    def evolution_of_the_cluster(self, cluster):
        '''
        Function that makes de cluster evolution.
        input: cluster -> defined in an inertial frame (centered at the Galactic center)
        steps in this function:
        1. From cluster, another cluster is defined in a rotating frame
        2. The gravity code is initialized
        3. The stellar evolution is initialized
        4. the Galaxy model is constructed
        5. The Galaxy and the cluster in the rotating frame are coupled via the Rotating Bridge
        6. The evolution of the system is made
        7. the cluster properties are transformed back to the inertial frame 
        '''

        cluster_in_rotating_frame = self.creation_cluster_in_rotating_frame(cluster)

        # N body code
        epsilon = self.softening(cluster)
        convert_nbody = nbody_system.nbody_to_si(cluster.mass.sum(), cluster.virial_radius())

        gravity = Huayno(convert_nbody)
        gravity.parameters.timestep = self.dt_bridge / 3.
        gravity.particles.add_particles(cluster_in_rotating_frame)
        gravity.parameters.epsilon_squared = epsilon ** 2
        channel_from_gravity_to_rotating_cluster = gravity.particles.new_channel_to(cluster_in_rotating_frame)
        channel_from_rotating_cluster_to_gravity = cluster_in_rotating_frame.new_channel_to(gravity.particles)

        # stellar evolution code
        se = SeBa()
        se.particles.add_particles(cluster_in_rotating_frame)
        channel_from_rotating_cluster_to_se = cluster_in_rotating_frame.new_channel_to(se.particles)
        channel_from_se_to_rotating_cluster = se.particles.new_channel_to(cluster_in_rotating_frame)

        # Galaxy model and Rotating bridge
        MW = self.galactic_model()
        system = Rotating_Bridge(self.omega_system, timestep=self.dt_bridge, verbose=False, method=self.method)
        system.add_system(gravity, (MW,), False)
        system.add_system(MW, (), False)

        X = []
        Y = []
        Z = []
        VX = []
        VY = []
        VZ = []
        T = []
        l = []
        m = []
        r = []
        age = []
        tem = []
        st = []
        # Cluster evolution
        while (self.time <= self.t_end - self.dt_bridge / 2):
            self.time += self.dt_bridge

            system.evolve_model(self.time)
            se.evolve_model(self.time)

            channel_from_gravity_to_rotating_cluster.copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz'])
            channel_from_se_to_rotating_cluster.copy_attributes(
                ['mass', 'radius', 'luminosity', 'age', 'temperature', 'stellar_type'])
            channel_from_rotating_cluster_to_gravity.copy_attributes(['mass'])
            self.from_noinertial_to_cluster_in_inertial_frame(cluster_in_rotating_frame, cluster)

            time = self.time.value_in(units.Myr)
            cm = cluster.center_of_mass()
            # print radius
            # write data
            if ((time == 2) or (time == 50) or (time == 100) or (time == 150)):
                X.append((cluster.x - cm[0]).value_in(units.kpc))
                Y.append((cluster.y - cm[1]).value_in(units.kpc))
                Z.append((cluster.z - cm[2]).value_in(units.kpc))
                VX.append((cluster.vx).value_in(units.kms))
                VY.append((cluster.vy).value_in(units.kms))
                VZ.append((cluster.vz).value_in(units.kms))

                T.append(time)
                m.append((cluster.mass).value_in(units.MSun))
                l.append((cluster.luminosity).value_in(units.LSun))
                r.append((cluster.radius).value_in(units.RSun))
                age.append((cluster.age).value_in(units.Myr))
                tem.append((cluster.temperature).value_in(units.K))
                st.append(cluster.stellar_type)

        gravity.stop()
        se.stop()

        return dict(age=age, time=T, x=X, y=Y, z=Z, mass=m, luminosity=l, radius=r, temperature=tem, stellar_type=st,
                    vx=VX, vy=VY, vz=VZ)

def stars_to_gaia_observables(age, mass, ra, dec, distance, luminosity, radius, temperature, stellar_type, vx, vy, vz):
    """
    Return what Gaia would observe given a set of stars.

    :param age: np.array
        Ages in Myr
    :param mass: np.array
        Mass in MSun
    :param ra: np.array
        RA J2000 in degrees
    :param dec: np.array
        DEC J2000 in degrees
    :param distance: np.array
        distance in parsec
    :param luminosity: np.array
        Luminosity in LSun
    :param radius: np.array
        Radius in RSun
    :param temperature: np.array
        Temperatur in K
    :param stellar_type: np.array
        Stellar type String
    :param v_ra: np.array
        proper motion in RA arcsec/yr
    :param v_dec: np.array
        proper motion in DEC arcsec/yr
    :param v_rad: np.array
        radial velocity in km/s
    :return: dict with the items:
    <gain observable> : List of tuple (observable value, observable uncertainty)

    Note, if Gaia would not see a star then do not include it in the list.
    If Gaia would see a star but not be able to constrain some properties, then put a appropriate uncertainty on it.
    E.g. if Gaia sees a star but can't determine the radial velocity, then the tuple for that V_rad is (0, 5000km/s).
    The large uncertainty should be the prior uncertainty that tells the machine that you don't know, but that you know stars have
    v_rad somewhere in +- 5000km/s.
    Likewise, if you don't know prop_ra/dec then put (0, large prior uncertainty).
    """
    return dict(age=age, mass=mass, ra=ra, dec=dec, distance=distance, luminosity=luminosity, radius=radius,
                temperature=temperature, stellar_type=stellar_type, vx=vx, vy=vy, vz=vz)



def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    optional = parser._action_groups.pop()  # Edited this line
    parser._action_groups.append(optional)  # added this line
    required = parser.add_argument_group('Required arguments')

    # network
    required.add_argument("--num_train", type=int, default=10,
                          help="""The number of clusters to simulate for training\n""")
    optional.add_argument("--num_test", type=int, default=None,
                          help="""The number of clusters to simulate for test\n""")
    optional.add_argument("--test_fraction", type=float, default=0.5,
                          help="""The fraction of all simulated clusters to use for testing.\n""")
    optional.add_argument("--time_limit", type=float, default=None,
                          help="""The number of seconds to allow simulations.\n""")
    required.add_argument("--output_folder", type=str, default=None,
                          help="""The output root directory.\n""")
    optional.add_argument("--num_processes", type=int, default=1,
                          help="""The number of processes to run in parallel (no time limit imposed).\n""")


def main(num_train, num_test, test_fraction, time_limit, output_folder, num_processes):
    if test_fraction is None:
        if num_test is None:
            raise ValueError("num_test must be int > 0 if test_fraction is None.")
        test_fraction = num_test / (num_train + num_test)

    total_desired = num_train / (1. - test_fraction)
    logging.info("Total desired examples: {}".format(total_desired))
    logging.info("Test fraction: {}".format(test_fraction))

    t0 = default_timer()
    if time_limit is None:
        time_limit = np.inf
    logging.info("Simulating with timelimit {} seconds".format(time_limit))

    output_folder = os.path.abspath(output_folder)
    train_folder = os.path.join(output_folder, 'train_examples')
    test_folder = os.path.join(output_folder, 'test_examples')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    logging.info("Using output folder {}".format(output_folder))

    num_done = 0
    while default_timer() - t0 < time_limit and num_done < total_desired:
        logging.info("Working on example: {:04d}".format(num_done))
        # Create a cluster
        star_cluster = create_cluster_with_IMF(N=200)
        star_cluster.position += [-6.5, 0, 0] | units.kpc
        star_cluster.velocity += [0, 50, 0] | units.kms

        # Construct the galactic model and make the evolution
        evolution = RealisticEvolutionCluster(simulation_time=150 | units.Myr)
        res_dict = evolution.evolution_of_the_cluster(star_cluster)

        gaia_dict = stars_to_gaia_observables(**res_dict)

        save_location = os.path.join(train_folder, "example_{:05d}".format(num_done))

        # plots
        plot_location = os.path.join(save_location, 'data_plots')
        os.makedirs(plot_location, exist_ok=True)

        for i in range(len(res_dict.keys())):
            key1 = res_dict.keys()[i]
            for j in range(i+1, len(res_dict.keys())):
                key2 = res_dict.keys()[j]
                plt.figure(figsize=(6,6))
                plt.scatter(res_dict[key1],res_dict[key2], c='black', marker='+', alpha=0.5)
                points = np.stack([res_dict[key1],res_dict[key2]] ,axis=0)
                kernel = st.gaussian_kde(points)
                z = kernel(points)
                plt.tricontour(res_dict[key1],res_dict[key2],z, levels=7, linewidths=0.5, colors='k')
                plt.xlabel(key1)
                plt.ylabel(key2)
                plt.grid()
                plt.tight_layout()

                plt.savefig(os.path.join(plot_location,'{}_{}.png'.format(key1, key2)))
                plt.close('all')
        with h5py.File(os.path.join(save_location, "starcluster_data.h5"), 'w') as f:
            for key, value in res_dict.items():
                f[key] = np.array(value)

        with h5py.File(os.path.join(save_location, "gaia_data.h5"), 'w') as f:
            for key, value in gaia_dict.items():
                f[key] = np.array(value)


if __name__ in ('__main__'):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logging.info("Running with:")
    for option, value in vars(flags).items():
        logging.info("    {} -> {}".format(option, value))
    main(**vars(flags))
