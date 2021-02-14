/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  
  // construct normal distribution before loop (mu and std are the same for all the particles)
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle_sample = {};
    particle_sample.x = dist_x(gen);
    particle_sample.y = dist_y(gen);
    particle_sample.theta = dist_theta(gen);
    particle_sample.weight = 1.0;
    particles.push_back(particle_sample);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; ++i) {
    double x_pred;
    double y_pred;
    double theta_pred;
    
    double x_pred_sinpart = sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta);
    double y_pred_cospart = cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t);
    
    if (fabs(yaw_rate) < 0.0001) {
      x_pred = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y_pred = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }
    else {
      x_pred = particles[i].x + velocity * x_pred_sinpart / yaw_rate;
      y_pred = particles[i].y + velocity * y_pred_cospart / yaw_rate;
    }
    theta_pred = particles[i].theta + yaw_rate * delta_t;
    
    // construct normal distribution inside loop (mu is different for each particle)
    std::normal_distribution<double> dist_x(x_pred, std_pos[0]);
    std::normal_distribution<double> dist_y(y_pred, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_pred, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // to be modified after finishing updateWeights()
  // guess we want to find the closest landmark and assign the landmark id to the observation
  double min_dist = 0.0;
  double new_dist;
  int associated_id;
  
  for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx){
    min_dist = dist(observations[obs_idx].x, observations[obs_idx].y, predicted[0].x, predicted[0].y);
    associated_id = predicted[0].id;
    
    new_dist = min_dist + 1.0;  // keep new_dist always greater than min_dist when initializing
    for (int landmark_idx = 1; landmark_idx < predicted.size(); ++landmark_idx){
      new_dist = dist(observations[obs_idx].x, observations[obs_idx].y, predicted[landmark_idx].x, predicted[landmark_idx].y);
      if (new_dist < min_dist) {
        min_dist = new_dist;
        associated_id = predicted[landmark_idx].id;
      }
    }
    
    observations[obs_idx].id = associated_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double obs_x_map;
  double obs_y_map;
  
  for (int i = 0; i < num_particles; ++i) {
    
    // transform obs from vehicle coord sys to map coord sys for each particle
    vector<LandmarkObs> obs_map_vectors;
    for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
      obs_x_map = particles[i].x + cos(particles[i].theta) * observations[obs_idx].x - sin(particles[i].theta) *  observations[obs_idx].y;
      obs_y_map = particles[i].y + sin(particles[i].theta) * observations[obs_idx].x + cos(particles[i].theta) *  observations[obs_idx].y;
      
      LandmarkObs obs_map_entry;
      obs_map_entry.x = obs_x_map;
      obs_map_entry.y = obs_y_map;
      obs_map_vectors.push_back(obs_map_entry);
    }
    
    // filtering landmarks within sensor range
    vector<LandmarkObs> temp_landmark_vectors;
    for (int lm_idx = 0; lm_idx < map_landmarks.landmark_list.size(); ++lm_idx) {
      if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[lm_idx].x_f, map_landmarks.landmark_list[lm_idx].y_f) <= sensor_range) {
        LandmarkObs temp_landmark_entry;
        temp_landmark_entry.id = map_landmarks.landmark_list[lm_idx].id_i;
        temp_landmark_entry.x = map_landmarks.landmark_list[lm_idx].x_f;
        temp_landmark_entry.y = map_landmarks.landmark_list[lm_idx].y_f;
        temp_landmark_vectors.push_back(temp_landmark_entry);
      }
    }
    
    // associate observations with landmarks (in map coord sys)
    // TODO: check the size of temp_landmark_vectors. if none of landmarks is within the sensor range, weight shall be 0
    if (temp_landmark_vectors.size() > 0) {
      dataAssociation(temp_landmark_vectors, obs_map_vectors);
      
      // maybe set associations to paritcle's attributes
      vector<int> assoid_vector;
      vector<double> obs_map_xvector;
      vector<double> obs_map_yvector;
      for (int obs_idx = 0; obs_idx < obs_map_vectors.size(); ++obs_idx) {
        assoid_vector.push_back(obs_map_vectors[obs_idx].id);
        obs_map_xvector.push_back(obs_map_vectors[obs_idx].x);
        obs_map_yvector.push_back(obs_map_vectors[obs_idx].y);
        
        // tricky way to change values in obs_map_vectors with landmark pos
        int lm_idx = 0;
        while (temp_landmark_vectors[lm_idx].id != obs_map_vectors[obs_idx].id) {
          ++lm_idx;
        }
        obs_map_vectors[obs_idx].x = temp_landmark_vectors[lm_idx].x;
        obs_map_vectors[obs_idx].y = temp_landmark_vectors[lm_idx].y;
      }
      SetAssociations(particles[i], assoid_vector, obs_map_xvector, obs_map_yvector);
      
      // calculate weight of a certain particle based on Multi-variate Gaussian multiplied over all measurements
      double exponent;
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      for (int obs_idx = 0; obs_idx < obs_map_vectors.size(); ++obs_idx) {
        exponent = (pow(particles[i].sense_x[obs_idx] - obs_map_vectors[obs_idx].x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(particles[i].sense_y[obs_idx] - obs_map_vectors[obs_idx].y, 2) / (2 * pow(std_landmark[1], 2)));
        particles[i].weight *= gauss_norm * exp(-exponent);
      }
    }
    else {
      //std::cout << particles[i].x << particles[i].y << std::endl;
      particles[i].weight = 0;
    }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // normalize weights
  double weight_sum = 0;
  double max_weight = 0;
  vector<double> weight_vector;
  for (int i = 0; i < num_particles; ++i) {
    weight_sum += particles[i].weight;
  }
  for (int i = 0; i < num_particles; ++i) {
    particles[i].weight /= weight_sum;
    if (max_weight < particles[i].weight) {
      max_weight = particles[i].weight;
    }
    weight_vector.push_back(particles[i].weight);
  }

  // resampling with discrete distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> d(weight_vector.begin(), weight_vector.end());
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; ++i) {
    Particle particle_newsample = particles[d(gen)];
    new_particles.push_back(particle_newsample);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
