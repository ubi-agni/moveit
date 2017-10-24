// Copyright  (C)  2007  Ruben Smits <ruben dot smits at mech dot kuleuven dot be>

// Version: 1.0
// Author: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>
// Maintainer: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>
// URL: http://www.orocos.org/kdl

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

// Modified to account for "mimic" joints, i.e. joints whose motion has a
// linear relationship to that of another joint.
// Copyright  (C)  2013  Sachin Chitta, Willow Garage

#include <moveit/kdl_kinematics_plugin/chainiksolver_vel_pinv_mimic.hpp>
#include <ros/console.h>

namespace KDL
{
ChainIkSolverVel_pinv_mimic::ChainIkSolverVel_pinv_mimic(const Chain& _chain, int _num_mimic_joints, int _num_redundant_joints,
                                                         double _position_weight, double _orientation_weight, bool _position_ik, double _eps)
  : chain(_chain)
  , jnt2jac(chain)
  , jac(chain.getNrOfJoints())
  , jac_reduced(chain.getNrOfJoints() - _num_mimic_joints)
  , jac_locked(chain.getNrOfJoints() - _num_redundant_joints - _num_mimic_joints)
  , svd(_position_ik ? 3 : 6, chain.getNrOfJoints() - _num_mimic_joints, Eigen::ComputeThinU | Eigen::ComputeThinV)
  , eps(_eps)
  , num_mimic_joints(_num_mimic_joints)
  , position_ik(_position_ik)
  , num_redundant_joints(_num_redundant_joints)
  , redundant_joints_locked(false)
  , position_weight_(1.0)
  , orientation_weight_(1.0)
{
  mimic_joints_.resize(chain.getNrOfJoints());
  for (std::size_t i = 0; i < mimic_joints_.size(); ++i)
    mimic_joints_[i].reset(i);
}

void ChainIkSolverVel_pinv_mimic::updateInternalDataStructures()
{
  // TODO: move (re)allocation of any internal data structures here
  // to react to changes in chain
}

ChainIkSolverVel_pinv_mimic::~ChainIkSolverVel_pinv_mimic()
{
}

bool ChainIkSolverVel_pinv_mimic::setMimicJoints(const std::vector<kdl_kinematics_plugin::JointMimic>& mimic_joints)
{
  if (mimic_joints.size() != chain.getNrOfJoints())
    return false;

  for (std::size_t i = 0; i < mimic_joints.size(); ++i)
  {
    if (mimic_joints[i].map_index >= chain.getNrOfJoints())
      return false;
  }
  mimic_joints_ = mimic_joints;
  return true;
}

bool ChainIkSolverVel_pinv_mimic::setRedundantJointsMapIndex(
    const std::vector<unsigned int>& redundant_joints_map_index)
{
  if (redundant_joints_map_index.size() != chain.getNrOfJoints() - num_mimic_joints - num_redundant_joints)
  {
    ROS_ERROR("Map index size: %d does not match expected size. "
              "No. of joints: %d, num_mimic_joints: %d, num_redundant_joints: %d",
              (int)redundant_joints_map_index.size(), (int)chain.getNrOfJoints(), (int)num_mimic_joints,
              (int)num_redundant_joints);
    return false;
  }

  for (std::size_t i = 0; i < redundant_joints_map_index.size(); ++i)
  {
    if (redundant_joints_map_index[i] >= chain.getNrOfJoints() - num_mimic_joints)
      return false;
  }
  locked_joints_map_index = redundant_joints_map_index;
  return true;
}

bool ChainIkSolverVel_pinv_mimic::jacToJacReduced(const Jacobian& jac, Jacobian& jac_reduced_l)
{
  jac_reduced_l.data.setZero();
  for (std::size_t i = 0; i < chain.getNrOfJoints(); ++i)
  {
    Twist vel1 = jac_reduced_l.getColumn(mimic_joints_[i].map_index);
    Twist vel2 = jac.getColumn(i);
    Twist result = vel1 + (mimic_joints_[i].multiplier * vel2);
    jac_reduced_l.setColumn(mimic_joints_[i].map_index, result);
  }
  return true;
}

bool ChainIkSolverVel_pinv_mimic::weightJac(Jacobian& jac)
{
  for (std::size_t i = 0; i < jac.columns(); ++i)
  {
    assert(mimic_joints_[i].map_index == i);
    if (mimic_joints_[i].weight != 1.0)
      jac.data.col(i) *= mimic_joints_[i].weight;
  }
  if (position_weight_ != 1.0)
    jac.data.topRows<3>() *= position_weight_;
  if (orientation_weight_ != 1.0)
    jac.data.bottomRows<3>() *= orientation_weight_;
  return true;
}

bool ChainIkSolverVel_pinv_mimic::jacToJacLocked(const Jacobian& jac, Jacobian& jac_locked)
{
  jac_locked.data.setZero();
  for (std::size_t i = 0; i < chain.getNrOfJoints() - num_mimic_joints - num_redundant_joints; ++i)
  {
    jac_locked.setColumn(i, jac.getColumn(locked_joints_map_index[i]));
  }
  return true;
}

// compute q_out = W_j * (W_x * J * W_j)^# * W_x * v_in
// where W_j and W_x are joint and task-level weights
int ChainIkSolverVel_pinv_mimic::CartToJnt(const JntArray& q_in, const Twist& v_in, JntArray& qdot_out)
{
  // Let the ChainJntToJacSolver calculate the jacobian "jac" for
  // the current joint positions "q_in". This will include the mimic joints
  if (num_mimic_joints > 0)
  {
    jnt2jac.JntToJac(q_in, jac);
    // Now compute the actual jacobian that involves only the active DOFs
    jacToJacReduced(jac, jac_reduced);
  }
  else
    jnt2jac.JntToJac(q_in, jac_reduced);

  // apply weighting to Jacobian
  weightJac(jac_reduced);

  // transform v_in to 6D Eigen::Vector and apply weigthing
  Eigen::Matrix<double, 6, 1> vin;
  vin.topRows<3>() = position_weight_ * Eigen::Map<const Eigen::Vector3d>(v_in.vel.data, 3);
  vin.topRows<3>() = orientation_weight_ * Eigen::Map<const Eigen::Vector3d>(v_in.rot.data, 3);

  // Remove columns of locked redundant joints from Jacobian
  bool locked = (redundant_joints_locked && num_redundant_joints > 0);
  if (locked)
    jacToJacLocked(jac_reduced, jac_locked);

  // use jac_reduced or jac_locked in the following
  Eigen::MatrixXd J = locked ? jac_locked.data : jac_reduced.data;

  unsigned int columns = J.cols();
  unsigned int rows = position_ik ? 3 : J.rows();

  // Do a singular value decomposition of "jac" with maximum
  // iterations "maxiter", put the results in "U", "S" and "V"
  // jac = U*S*Vt

  svd.compute(J.topRows(rows));
  qdot_out.data.block(0,0, columns,1) = svd.solve(vin.topRows(rows));

  // apply joint-weighting
  for (int i = 0; i < columns; ++i)
  {
      qdot_out(i) *= mimic_joints_[i].weight;
  }

  ROS_DEBUG_STREAM_NAMED("kdl", "Solution:");
  if (num_mimic_joints > 0)
  {
    for (int i = columns; i < chain.getNrOfJoints(); ++i)
    {
      qdot_out(i) = qdot_out(mimic_joints_[i].map_index) * mimic_joints_[i].multiplier;
    }
  }
  return 0;
}
}
