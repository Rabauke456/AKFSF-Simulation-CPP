 // ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Extended Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double INIT_VEL_STD = 10.0;
constexpr double INIT_PSI_STD = 45.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: use the wrapAngle() function on angular values to always keep angle
        // values within correct range, otherwise strange angle effects might be seen.
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1)
        {           
            // The map matched beacon positions can be accessed using: map_beacon.x AND map_beacon.y
            // calculate measurement vector z from GPS, Gyro and Lidar Data x_hat that was calculated in the funcatino predictionStep
            double delta_x = map_beacon.x - state[0];
            double delta_y = map_beacon.y - state[1];
            double r_hat = std::sqrt(std::pow(delta_x, 2) + std::pow(delta_y, 2));
            double theta_hat = wrapAngle(std::atan2(delta_y, delta_x) - state(2));
            Vector2d z_hat_measurement_prediction;
            z_hat_measurement_prediction << r_hat, theta_hat;
            
            // Calculate the measurement Innovation
            // Determine the innovation vector
            Vector2d z_lidar_measurement;
            z_lidar_measurement << meas.range,meas.theta;
            Vector2d nu_innovation = z_lidar_measurement - z_hat_measurement_prediction;
            nu_innovation(1) = wrapAngle(nu_innovation(1));
            // std::cout << "Nu Vector: " << nu_innovation.transpose() << std::endl;
            
            // Calculate the Measurement Jacobian H, Model derivative
            MatrixXd H = MatrixXd::Zero(2,4);
            double d = std::sqrt(std::pow(delta_x, 2) + std::pow(delta_y, 2));
            H(0,0) = -delta_x/d;
            H(0,1) = -delta_y/d;
            H(1,0) = delta_y/(std::pow(d, 2));
            H(1,1) = -delta_x/(std::pow(d, 2));
            H(1,2) = -1;
            //std::cout << "Measurement Jacobian H: " << H << std::endl;


            // Measurement Noise Covariance S, calculated uncertainty of the Innovation vector nu
            Matrix2d R = Matrix2d::Zero();
            R(0,0) = std::pow(LIDAR_RANGE_STD, 2);
            R(1,1) = std::pow(LIDAR_THETA_STD, 2);
            Matrix2d S = H * cov * H.transpose() + R;

            // Determine the Kalman Gain K, should be smaller than 1, the more certain the measurement is
            MatrixXd K = cov * H.transpose() * S.inverse();
            // std::cout << "Kalman Gain K: " << K << std::endl;

            // Update State and Covariance
            state = state + K * nu_innovation;
            cov = (Matrix4d::Identity() - K * H) * cov;

        }

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: use the wrapAngle() function on angular values to always keep angle
        // values within correct range, otherwise strange angle effects might be seen.
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        // Implement Prediction Step
        state(0) += dt*state(3)*cos((state(2)));
        state(1) += dt*state(3)*sin((state(2)));
        state(2) = wrapAngle(state(2) + gyro.psi_dot*dt);
        //state(3) += 1/2; // No acceleration input available, set velocity to zero or maintain previous velocity

        // Jacobian F
        Matrix4d F = Matrix4d::Identity();
        F(0,2) = -dt*state(3)*sin((state(2)));
        F(0,3) = dt*cos((state(2)));
        F(1,2) = dt*state(3)*cos((state(2)));
        F(1,3) = dt*sin((state(2)));

        // Process Noise Covariance Q
        Matrix4d Q = Matrix4d::Zero();
        Q(2,2) = std::pow(dt, 2)*std::pow(GYRO_STD, 2);
        Q(3,3) = std::pow(dt, 2)*std::pow(ACCEL_STD, 2);

        // Covariance Prediction
        cov = F*cov*F.transpose() + Q;


        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the EKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,4);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H << 1,0,0,0,0,1,0,0;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (Matrix4d::Identity() - K*H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        VectorXd state = Vector4d::Zero();
        MatrixXd cov = Matrix4d::Zero();

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;

        setState(state);
        setCovariance(cov);
    } 
             
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}
