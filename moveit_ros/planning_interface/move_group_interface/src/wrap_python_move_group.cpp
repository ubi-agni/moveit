/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Ioan Sucan */

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/py_bindings_tools/roscpp_initializer.h>
#include <moveit/py_bindings_tools/ros_msg_typecasters.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/trajectory_processing/iterative_spline_parameterization.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <memory>
#include <Python.h>

/** @cond IGNORE */

namespace py = pybind11;

namespace moveit
{
namespace planning_interface
{
struct deserializionError : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

class MoveGroupInterfaceWrapper : protected py_bindings_tools::ROScppInitializer, public MoveGroupInterface
{
public:
  // ROSInitializer is constructed first, and ensures ros::init() was called, if
  // needed
  MoveGroupInterfaceWrapper(const std::string& group_name, const std::string& robot_description,
                            const std::string& ns = "", double wait_for_servers = 5.0)
    : py_bindings_tools::ROScppInitializer()
    , MoveGroupInterface(Options(group_name, robot_description, ros::NodeHandle(ns)),
                         std::shared_ptr<tf2_ros::Buffer>(), ros::WallDuration(wait_for_servers))
  {
  }

  bool setJointValueTargetFromPosePython(const geometry_msgs::Pose& pose_msg, const std::string& eef, bool approx)
  {
    return approx ? setApproximateJointValueTarget(pose_msg, eef) : setJointValueTarget(pose_msg, eef);
  }

  bool setJointValueTargetFromPoseStampedPython(const geometry_msgs::PoseStamped& pose_msg, const std::string& eef,
                                                bool approx)
  {
    return approx ? setApproximateJointValueTarget(pose_msg, eef) : setJointValueTarget(pose_msg, eef);
  }

  std::vector<double> getJointValueTargetPythonList()
  {
    std::vector<double> values;
    MoveGroupInterface::getJointValueTarget(values);
    return values;
  }

  void rememberJointValuesFromPythonList(const std::string& string, const std::vector<double>& values)
  {
    rememberJointValues(string, values);
  }

  const char* getPlanningFrameCStr() const
  {
    return getPlanningFrame().c_str();
  }

  moveit_msgs::PlannerInterfaceDescription getInterfaceDescriptionPython()
  {
    moveit_msgs::PlannerInterfaceDescription msg;
    getInterfaceDescription(msg);
    return msg;
  }

  bool placePose(const std::string& object_name, geometry_msgs::Pose pose, bool plan_only = false)
  {
    geometry_msgs::PoseStamped msg;
    msg.pose = std::move(pose);
    msg.header.frame_id = getPoseReferenceFrame();
    msg.header.stamp = ros::Time::now();
    py::gil_scoped_release gr;
    return place(object_name, msg, plan_only) == MoveItErrorCode::SUCCESS;
  }

  bool placePoses(const std::string& object_name, std::vector<geometry_msgs::PoseStamped> const& poses_list,
                  bool plan_only = false)
  {
    py::gil_scoped_release gr;
    return place(object_name, poses_list, plan_only) == MoveItErrorCode::SUCCESS;
  }

  bool placeLocations(const std::string& object_name, std::vector<moveit_msgs::PlaceLocation> location_list,
                      bool plan_only = false)
  {
    py::gil_scoped_release gr;
    return place(object_name, std::move(location_list), plan_only) == MoveItErrorCode::SUCCESS;
  }

  bool placeAnywhere(const std::string& object_name, bool plan_only = false)
  {
    py::gil_scoped_release gr;
    return place(object_name, plan_only) == MoveItErrorCode::SUCCESS;
  }

  moveit_msgs::RobotState getCurrentStateBoundedPython()
  {
    moveit::core::RobotStatePtr current = getCurrentState();
    current->enforceBounds();
    moveit_msgs::RobotState rsmv;
    moveit::core::robotStateToRobotStateMsg(*current, rsmv);
    return rsmv;
  }

  moveit_msgs::RobotState getCurrentStatePython()
  {
    moveit::core::RobotStatePtr current_state = getCurrentState();
    moveit_msgs::RobotState state_message;
    moveit::core::robotStateToRobotStateMsg(*current_state, state_message);
    return state_message;
  }

  const char* getEndEffectorLinkCStr() const
  {
    return getEndEffectorLink().c_str();
  }

  const char* getPoseReferenceFrameCStr() const
  {
    return getPoseReferenceFrame().c_str();
  }

  const char* getNameCStr() const
  {
    return getName().c_str();
  }

  const char* getPlannerIdCStr() const
  {
    return getPlannerId().c_str();
  }

  const char* getPlanningPipelineIdCStr() const
  {
    return getPlanningPipelineId().c_str();
  }

  bool movePython()
  {
    py::gil_scoped_release gr;
    return move() == MoveItErrorCode::SUCCESS;
  }

  bool asyncMovePython()
  {
    return asyncMove() == MoveItErrorCode::SUCCESS;
  }

  bool executePython(const moveit_msgs::RobotTrajectory& plan)
  {
    py::gil_scoped_release gr;
    return execute(plan) == MoveItErrorCode::SUCCESS;
  }

  bool asyncExecutePython(const moveit_msgs::RobotTrajectory& plan)
  {
    return asyncExecute(plan) == MoveItErrorCode::SUCCESS;
  }

  std::tuple<moveit_msgs::MoveItErrorCodes, moveit_msgs::RobotTrajectory, double> planPython()
  {
    MoveGroupInterface::Plan plan;
    moveit_msgs::MoveItErrorCodes res;
    py::gil_scoped_release gr;
    res = MoveGroupInterface::plan(plan);
    return { res, plan.trajectory_, plan.planning_time_ };
  }

  moveit_msgs::MotionPlanRequest constructMotionPlanRequestPython()
  {
    moveit_msgs::MotionPlanRequest request;
    constructMotionPlanRequest(request);
    return request;
  }

  std::tuple<moveit_msgs::RobotTrajectory, double>
  computeCartesianPathPython(const std::vector<geometry_msgs::Pose>& waypoints, double eef_step, double jump_threshold,
                             bool avoid_collisions)
  {
    return computeCartesianPathConstrainedPython(waypoints, eef_step, jump_threshold, avoid_collisions, {});
  }

  std::tuple<moveit_msgs::RobotTrajectory, double>
  computeCartesianPathConstrainedPython(const std::vector<geometry_msgs::Pose>& poses, double eef_step,
                                        double jump_threshold, bool avoid_collisions,
                                        const moveit_msgs::Constraints& path_constraints)
  {
    moveit_msgs::RobotTrajectory trajectory;
    double fraction;
    py::gil_scoped_release gr;
    fraction = computeCartesianPath(poses, eef_step, jump_threshold, trajectory, path_constraints, avoid_collisions);
    return { trajectory, fraction };
  }

  moveit_msgs::RobotTrajectory retimeTrajectory(const moveit_msgs::RobotState& ref_state_msg,
                                                const moveit_msgs::RobotTrajectory& traj_msg,
                                                double velocity_scaling_factor, double acceleration_scaling_factor,
                                                const std::string& algorithm)
  {
    py::gil_scoped_release gr;
    // Convert reference state message to object
    moveit::core::RobotState ref_state_obj(getRobotModel());
    if (!moveit::core::robotStateMsgToRobotState(ref_state_msg, ref_state_obj, true))
    {
      ROS_ERROR("Unable to convert RobotState message to RobotState instance.");
      throw deserializionError("Unable to convert RobotState message to RobotState instance.");
    }

    // Convert trajectory message to object
    robot_trajectory::RobotTrajectory traj_obj(getRobotModel(), getName());
    traj_obj.setRobotTrajectoryMsg(ref_state_obj, traj_msg);

    // Do the actual retiming
    if (algorithm == "iterative_time_parameterization")
    {
      trajectory_processing::IterativeParabolicTimeParameterization time_param;
      time_param.computeTimeStamps(traj_obj, velocity_scaling_factor, acceleration_scaling_factor);
    }
    else if (algorithm == "iterative_spline_parameterization")
    {
      trajectory_processing::IterativeSplineParameterization time_param;
      time_param.computeTimeStamps(traj_obj, velocity_scaling_factor, acceleration_scaling_factor);
    }
    else if (algorithm == "time_optimal_trajectory_generation")
    {
      trajectory_processing::TimeOptimalTrajectoryGeneration time_param;
      time_param.computeTimeStamps(traj_obj, velocity_scaling_factor, acceleration_scaling_factor);
    }
    else
    {
      ROS_ERROR_STREAM_NAMED("move_group_py", "Unknown time parameterization algorithm: " << algorithm);
      return {};
    }

    moveit_msgs::RobotTrajectory traj_msg_ans;
    // Convert the retimed trajectory back into a message
    traj_obj.getRobotTrajectoryMsg(traj_msg_ans);
    return traj_msg_ans;
  }

  Eigen::MatrixXd getJacobianMatrixPython(const std::vector<double>& joint_values,
                                          const std::array<double, 3>& reference_point)
  {
    moveit::core::RobotState state(getRobotModel());
    state.setToDefaultValues();
    auto group = state.getJointModelGroup(getName());
    state.setJointGroupPositions(group, joint_values);
    return state.getJacobian(group, Eigen::Map<const Eigen::Vector3d>(&reference_point[0]));
  }

  moveit_msgs::RobotState enforceBoundsPython(const moveit_msgs::RobotState& state_msg)
  {
    moveit::core::RobotState state(getRobotModel());
    if (moveit::core::robotStateMsgToRobotState(state_msg, state, true))
    {
      state.enforceBounds();
      moveit_msgs::RobotState ans;
      moveit::core::robotStateToRobotStateMsg(state, ans);
      return ans;
    }
    else
    {
      ROS_ERROR("Unable to convert RobotState message to RobotState instance.");
      throw deserializionError("Unable to convert RobotState message to RobotState instance.");
    }
  }
};
}  // namespace planning_interface
}  // namespace moveit

PYBIND11_MODULE(_moveit_move_group_interface, m)
{
  py::register_exception<moveit::planning_interface::deserializionError>(m, "DeserializationError");

  using moveit::planning_interface::MoveGroupInterface;
  using moveit::planning_interface::MoveGroupInterfaceWrapper;

  py::class_<MoveGroupInterfaceWrapper> move_group_interface_class(m, "MoveGroupInterface");

  move_group_interface_class.def(py::init<std::string, std::string>());
  move_group_interface_class.def(py::init<std::string, std::string, std::string>());
  move_group_interface_class.def(py::init<std::string, std::string, std::string, double>());

  move_group_interface_class.def("async_move", &MoveGroupInterfaceWrapper::asyncMovePython);
  move_group_interface_class.def("move", &MoveGroupInterfaceWrapper::movePython);
  move_group_interface_class.def("execute", &MoveGroupInterfaceWrapper::executePython);
  move_group_interface_class.def("async_execute", &MoveGroupInterfaceWrapper::asyncExecutePython);

  move_group_interface_class.def(
      "pick", py::overload_cast<const std::string&, std::vector<moveit_msgs::Grasp>, bool>(&MoveGroupInterface::pick));
  move_group_interface_class.def("place", &MoveGroupInterfaceWrapper::placePose);
  move_group_interface_class.def("place_poses_list", &MoveGroupInterfaceWrapper::placePoses);
  move_group_interface_class.def("place_locations_list", &MoveGroupInterfaceWrapper::placeLocations);
  move_group_interface_class.def("place", &MoveGroupInterfaceWrapper::placeAnywhere);
  move_group_interface_class.def("stop", &MoveGroupInterfaceWrapper::stop);

  move_group_interface_class.def("get_name", &MoveGroupInterfaceWrapper::getNameCStr);
  move_group_interface_class.def("get_planning_frame", &MoveGroupInterfaceWrapper::getPlanningFrameCStr);
  move_group_interface_class.def("get_interface_description", &MoveGroupInterfaceWrapper::getInterfaceDescriptionPython);

  move_group_interface_class.def("get_active_joints", &MoveGroupInterface::getActiveJoints);
  move_group_interface_class.def("get_joints", &MoveGroupInterface::getJoints);
  move_group_interface_class.def("get_variable_count", &MoveGroupInterfaceWrapper::getVariableCount);
  move_group_interface_class.def("allow_looking", &MoveGroupInterfaceWrapper::allowLooking);
  move_group_interface_class.def("allow_replanning", &MoveGroupInterfaceWrapper::allowReplanning);

  move_group_interface_class.def("set_pose_reference_frame", &MoveGroupInterfaceWrapper::setPoseReferenceFrame);

  move_group_interface_class.def("set_pose_reference_frame", &MoveGroupInterfaceWrapper::setPoseReferenceFrame);
  move_group_interface_class.def("set_end_effector_link", &MoveGroupInterfaceWrapper::setEndEffectorLink);
  move_group_interface_class.def("get_end_effector_link", &MoveGroupInterfaceWrapper::getEndEffectorLinkCStr);
  move_group_interface_class.def("get_pose_reference_frame", &MoveGroupInterfaceWrapper::getPoseReferenceFrameCStr);

  // move_group_interface_class.def("set_pose_target", py::overload_cast<const geometry_msgs::PoseStamped&, const
  // std::string&>(&MoveGroupInterface::setPoseTarget));
  move_group_interface_class.def("set_pose_target", py::overload_cast<const geometry_msgs::Pose&, const std::string&>(
                                                        &MoveGroupInterface::setPoseTarget));
  move_group_interface_class.def("set_pose_targets",
                                 py::overload_cast<const std::vector<geometry_msgs::Pose>&, std::string const&>(
                                     &MoveGroupInterface::setPoseTargets));

  move_group_interface_class.def("set_position_target", &MoveGroupInterfaceWrapper::setPositionTarget);
  move_group_interface_class.def("set_rpy_target", &MoveGroupInterfaceWrapper::setRPYTarget);
  move_group_interface_class.def("set_orientation_target", &MoveGroupInterfaceWrapper::setOrientationTarget);

  move_group_interface_class.def("get_current_pose", &MoveGroupInterface::getCurrentPose);
  move_group_interface_class.def("get_current_rpy", &MoveGroupInterface::getCurrentRPY);

  move_group_interface_class.def("get_random_pose", &MoveGroupInterface::getRandomPose);

  move_group_interface_class.def("clear_pose_target", &MoveGroupInterfaceWrapper::clearPoseTarget);
  move_group_interface_class.def("clear_pose_targets", &MoveGroupInterfaceWrapper::clearPoseTargets);

  move_group_interface_class.def("set_joint_value_target", py::overload_cast<const std::vector<double>&>(
                                                               &MoveGroupInterface::setJointValueTarget));
  move_group_interface_class.def("set_joint_value_target", py::overload_cast<std::map<std::string, double> const&>(
                                                               &MoveGroupInterface::setJointValueTarget));

  move_group_interface_class.def(
      "set_joint_value_target",
      py::overload_cast<const std::string&, const std::vector<double>&>(&MoveGroupInterface::setJointValueTarget));
  move_group_interface_class.def("set_joint_value_target", py::overload_cast<const std::string&, double>(
                                                               &MoveGroupInterface::setJointValueTarget));

  move_group_interface_class.def("set_joint_value_target_from_pose",
                                 &MoveGroupInterfaceWrapper::setJointValueTargetFromPosePython);
  move_group_interface_class.def("set_joint_value_target_from_pose_stamped",
                                 &MoveGroupInterfaceWrapper::setJointValueTargetFromPoseStampedPython);
  move_group_interface_class.def("set_joint_value_target", py::overload_cast<sensor_msgs::JointState const&>(
                                                               &MoveGroupInterface::setJointValueTarget));

  move_group_interface_class.def("get_joint_value_target", &MoveGroupInterfaceWrapper::getJointValueTargetPythonList);

  move_group_interface_class.def("set_named_target", &MoveGroupInterfaceWrapper::setNamedTarget);
  move_group_interface_class.def("set_random_target", &MoveGroupInterfaceWrapper::setRandomTarget);

  move_group_interface_class.def("remember_joint_values",
                                 py::overload_cast<const std::string&>(&MoveGroupInterface::rememberJointValues));

  move_group_interface_class.def(
      "remember_joint_values",
      py::overload_cast<const std::string&, const std::vector<double>&>(&MoveGroupInterface::rememberJointValues));

  move_group_interface_class.def("start_state_monitor", &MoveGroupInterfaceWrapper::startStateMonitor);
  move_group_interface_class.def("get_current_joint_values", &MoveGroupInterface::getCurrentJointValues);
  move_group_interface_class.def("get_random_joint_values", &MoveGroupInterface::getRandomJointValues);
  move_group_interface_class.def("get_remembered_joint_values", &MoveGroupInterface::getRememberedJointValues);

  move_group_interface_class.def("forget_joint_values", &MoveGroupInterfaceWrapper::forgetJointValues);

  move_group_interface_class.def("get_goal_joint_tolerance", &MoveGroupInterfaceWrapper::getGoalJointTolerance);
  move_group_interface_class.def("get_goal_position_tolerance", &MoveGroupInterfaceWrapper::getGoalPositionTolerance);
  move_group_interface_class.def("get_goal_orientation_tolerance",
                                 &MoveGroupInterfaceWrapper::getGoalOrientationTolerance);

  move_group_interface_class.def("set_goal_joint_tolerance", &MoveGroupInterfaceWrapper::setGoalJointTolerance);
  move_group_interface_class.def("set_goal_position_tolerance", &MoveGroupInterfaceWrapper::setGoalPositionTolerance);
  move_group_interface_class.def("set_goal_orientation_tolerance",
                                 &MoveGroupInterfaceWrapper::setGoalOrientationTolerance);
  move_group_interface_class.def("set_goal_tolerance", &MoveGroupInterfaceWrapper::setGoalTolerance);

  move_group_interface_class.def("set_start_state_to_current_state",
                                 &MoveGroupInterfaceWrapper::setStartStateToCurrentState);
  move_group_interface_class.def("set_start_state",
                                 py::overload_cast<const moveit_msgs::RobotState&>(&MoveGroupInterface::setStartState));

  move_group_interface_class.def("set_path_constraints",
                                 py::overload_cast<const std::string&>(&MoveGroupInterface::setPathConstraints));
  move_group_interface_class.def("set_path_constraints", py::overload_cast<moveit_msgs::Constraints const&>(
                                                             &MoveGroupInterface::setPathConstraints));
  move_group_interface_class.def("get_path_constraints", &MoveGroupInterface::getPathConstraints);
  move_group_interface_class.def("clear_path_constraints", &MoveGroupInterfaceWrapper::clearPathConstraints);

  move_group_interface_class.def("set_trajectory_constraints", &MoveGroupInterface::setTrajectoryConstraints);
  move_group_interface_class.def("get_trajectory_constraints", &MoveGroupInterface::getTrajectoryConstraints);
  move_group_interface_class.def("clear_trajectory_constraints", &MoveGroupInterfaceWrapper::clearTrajectoryConstraints);
  move_group_interface_class.def("get_known_constraints", &MoveGroupInterface::getKnownConstraints);
  move_group_interface_class.def("set_constraints_database", &MoveGroupInterfaceWrapper::setConstraintsDatabase);
  move_group_interface_class.def("set_workspace", &MoveGroupInterfaceWrapper::setWorkspace);
  move_group_interface_class.def("set_planning_time", &MoveGroupInterfaceWrapper::setPlanningTime);
  move_group_interface_class.def("get_planning_time", &MoveGroupInterfaceWrapper::getPlanningTime);
  move_group_interface_class.def("set_max_velocity_scaling_factor",
                                 &MoveGroupInterfaceWrapper::setMaxVelocityScalingFactor);
  move_group_interface_class.def("set_max_acceleration_scaling_factor",
                                 &MoveGroupInterfaceWrapper::setMaxAccelerationScalingFactor);
  move_group_interface_class.def("set_planner_id", &MoveGroupInterfaceWrapper::setPlannerId);
  move_group_interface_class.def("get_planner_id", &MoveGroupInterfaceWrapper::getPlannerIdCStr);
  move_group_interface_class.def("set_planning_pipeline_id", &MoveGroupInterfaceWrapper::setPlanningPipelineId);
  move_group_interface_class.def("get_planning_pipeline_id", &MoveGroupInterfaceWrapper::getPlanningPipelineIdCStr);
  move_group_interface_class.def("set_num_planning_attempts", &MoveGroupInterfaceWrapper::setNumPlanningAttempts);
  move_group_interface_class.def("plan", &MoveGroupInterfaceWrapper::planPython);
  move_group_interface_class.def("construct_motion_plan_request",
                                 &MoveGroupInterfaceWrapper::constructMotionPlanRequestPython);
  move_group_interface_class.def("compute_cartesian_path", &MoveGroupInterfaceWrapper::computeCartesianPathPython);
  move_group_interface_class.def("compute_cartesian_path",
                                 &MoveGroupInterfaceWrapper::computeCartesianPathConstrainedPython);
  move_group_interface_class.def("set_support_surface_name", &MoveGroupInterfaceWrapper::setSupportSurfaceName);
  move_group_interface_class.def(
      "attach_object", py::overload_cast<const std::string&, const std::string&, const std::vector<std::string>&>(
                           &MoveGroupInterface::attachObject));
  move_group_interface_class.def("detach_object", &MoveGroupInterfaceWrapper::detachObject);
  move_group_interface_class.def("retime_trajectory", &MoveGroupInterfaceWrapper::retimeTrajectory);
  move_group_interface_class.def("get_named_targets", &MoveGroupInterface::getNamedTargets);
  move_group_interface_class.def("get_named_target_values", &MoveGroupInterface::getNamedTargetValues);
  move_group_interface_class.def("get_current_state_bounded", &MoveGroupInterfaceWrapper::getCurrentStateBoundedPython);
  move_group_interface_class.def("get_current_state", &MoveGroupInterfaceWrapper::getCurrentStatePython);
  move_group_interface_class.def("get_jacobian_matrix", &MoveGroupInterfaceWrapper::getJacobianMatrixPython,
                                 py::arg("joint_values"), py::arg("reference_point") = std::array<double, 3>{});
  move_group_interface_class.def("enforce_bounds", &MoveGroupInterfaceWrapper::enforceBoundsPython);
}

/** @endcond */
