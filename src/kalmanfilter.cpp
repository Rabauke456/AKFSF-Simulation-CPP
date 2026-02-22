// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Unscented Kalman Filter
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
constexpr double bias_std = GYRO_STD/10;
constexpr bool activate_sensor_bias_correction = true;
constexpr bool activate_non_zero_initial_condition = true;
constexpr bool activate_faulty_gps_measurement_correction = true;
// -------------------------------------------------- //

// ----------------------------------------------------------------------- //
// USEFUL HELPER FUNCTIONS

bool KalmanFilter::fully_initialised(std::vector<bool> init_vector) const
{
    bool v = true;
    bool allTrue = std::all_of(init_vector.begin(), init_vector.end(), [](bool v) { 
    return v;});
}

std::vector<bool> initialization_vector(VectorXd state)
{
    std::vector<bool> initialization_vector ;
    for(int i = 0; i < state.size(); i++)
    {
        if(state(i) != 0)
        {
            initialization_vector.push_back(true);
        }
        else
        {
            initialization_vector.push_back(false);
        }
    }
    return initialization_vector;
}

VectorXd normaliseState(VectorXd state)
{
    state(2) = wrapAngle(state(2));
    return state;
}
VectorXd normaliseLidarMeasurement(VectorXd meas)
{
    meas(1) = wrapAngle(meas(1));
    return meas;
}

bool innovation_in_bound (VectorXd innovation, MatrixXd S)
{
    // checks if the innovation passes the chi square test
    // TODO: get chi square values from function
    bool result = true;

    double normalised_innovation_squared;
    normalised_innovation_squared = innovation.transpose()*S.inverse()*innovation;

    double p_value = 0.05;
    int degrees_of_freedom = innovation.size();
    double chi_squared_distribution_value = 5.99;

    if(normalised_innovation_squared < std::pow(chi_squared_distribution_value, 2))
    {
        result = true;
    }
    else
    {
        result = false;
    }

    if(activate_faulty_gps_measurement_correction == false)
    {
        result = true;
    }
    else
    {
        result = result;
    }

    return result;
}

VectorXd gyro_bias(VectorXd model, double dt, VectorXd aug_state)
{
    int additional_columns = 1;
    int special_information_bias = 2; 

    VectorXd new_model = VectorXd::Zero(model.size() + additional_columns);
    double bias_wb = new_model(new_model.size()-1);
    new_model.head(model.size()) = model;
    new_model(special_information_bias) -= dt * bias_wb;

    //std::cout << "New Model with Gyro Bias: " << new_model.transpose() << std::endl;

    return new_model;
}

std::vector<VectorXd> generateSigmaPoints(VectorXd state, MatrixXd cov)
{
    std::vector<VectorXd> sigmaPoints;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    int n = state.size();
    double kappa = 3.0 - n;
    sigmaPoints.push_back(state);

    // calculate square root of covariance matrix
    MatrixXd square_root_covariance = cov.llt().matrixL();
    MatrixXd square_root_argument = std::sqrt(n + kappa)* square_root_covariance;

    for(unsigned int i = 0; i < n; i++){
            sigmaPoints.push_back(state + square_root_argument.col(i));
            sigmaPoints.push_back(state - square_root_argument.col(i));
    }

    // std::cout << "Sigma Point " << sigmaPoints << std::endl;

    // ----------------------------------------------------------------------- //

    return sigmaPoints;
}

std::vector<double> generateSigmaWeights(unsigned int numStates)
{
    std::vector<double> weights;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE

    unsigned int n = numStates;
    double kappa = 3.0 - n;
    weights.push_back(kappa / (n + kappa));

    for(unsigned int i = 0; i < 2*n; ++i){
        weights.push_back(0.5/ (n + kappa));
    }

    // std::cout << "Weight " << weights << std::endl;

    // ----------------------------------------------------------------------- //

    return weights;
}

VectorXd lidarMeasurementModel(VectorXd aug_state, double beaconX, double beaconY)
{
    VectorXd z_hat_mean_transformed_sigma_points = VectorXd::Zero(2);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    // Extract variables
    double px = aug_state(0);
    double py = aug_state(1);
    double psi = aug_state(2);
    double V = aug_state(3);
    double nu_r = aug_state(4);
    double nu_theta = aug_state(5);

    // calculate range and bearing
    double dx = beaconX - px;
    double dy = beaconY - py;

    // calculate components of measurement model
    double r_hat, theta_hat;
    r_hat = sqrt(dx*dx + dy*dy) + nu_r;
    theta_hat = atan2(dy, dx) - psi + nu_theta;

    z_hat_mean_transformed_sigma_points << r_hat, theta_hat;

    // ----------------------------------------------------------------------- //

    return z_hat_mean_transformed_sigma_points;
}

VectorXd vehicleProcessModel(VectorXd aug_state, double psi_dot, double dt)
{
    VectorXd new_state = VectorXd::Zero(4);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    new_state(0) = aug_state(0) + dt * aug_state(3) * cos(aug_state(2));
    new_state(1) = aug_state(1) + dt * aug_state(3) * sin(aug_state(2));
    new_state(2) = aug_state(2) + dt * (psi_dot + aug_state(4));
    new_state(3) = aug_state(3) + dt * aug_state(5);  

    if(activate_sensor_bias_correction == true)
    {
        new_state = gyro_bias(new_state, dt, aug_state);
    }

    //std::cout << "New State: " << new_state.transpose() << std::endl;

    // ----------------------------------------------------------------------- //

    return new_state;
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: Use the normaliseState() and normaliseLidarMeasurement() functions
        // to always keep angle values within correct range.
        // HINT: Do not normalise during sigma point calculation!
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1) // Check that we have a valid beacon match
        {
            // Augment the State and Covariance with Measurement Noise
            int additional_columns = 2;
            int n_model_size = state.size();
            int n_z = 2;

            // Generate Measurement Model Noise Covariance Matrix
            MatrixXd R = Matrix2d::Zero();
            R(0,0) = LIDAR_RANGE_STD*LIDAR_RANGE_STD;
            R(1,1) = LIDAR_THETA_STD*LIDAR_THETA_STD;

            // augmented state vector
            VectorXd x_aug = VectorXd::Zero(n_model_size + additional_columns);
            x_aug.head(n_model_size) = state;
            // std::cout << "Augmented State size: " << x_aug.size() << std::endl;
            // augmented covariance matrix
            MatrixXd P_aug = MatrixXd::Zero(n_model_size + additional_columns, n_model_size + additional_columns);
            P_aug.topLeftCorner(n_model_size,n_model_size) = cov;
            P_aug.bottomRightCorner(additional_columns,additional_columns) = R;
            // std::cout << "Augmented Covariance size: " << P_aug.rows() << "x" << P_aug.cols() << std::endl;

            // Generate Sigma Points and Weights
            std::vector<VectorXd> sigma_points = generateSigmaPoints(x_aug, P_aug);
            // std::cout << "Number of Sigma Points: " << sigma_points.size() << std::endl;
            std::vector<double> weights = generateSigmaWeights(n_model_size + additional_columns);
            // std::cout << "Number of Weights: " << weights.size() << std::endl;

            // Predict Measurement Sigma Points through Measurement Model
            std::vector<VectorXd> z_hat_mean_transformed_sigma_points;
            for (const auto& sigma_point : sigma_points)
            {
                z_hat_mean_transformed_sigma_points.push_back(lidarMeasurementModel(sigma_point, map_beacon.x, map_beacon.y));
            }
            // std::cout << "Dimensions of Sigma Points: " << z_hat_mean_transformed_sigma_points[0].size() << std::endl;

            // Predict Measurement Mean
            VectorXd z_mean = VectorXd::Zero(n_z);

            for(unsigned int i = 0; i < z_hat_mean_transformed_sigma_points.size(); ++i)
            {
                z_mean += weights[i] * z_hat_mean_transformed_sigma_points[i];
            }

            // std::cout << "z_mean size: " << z_mean.size() << std::endl;

            // Calculate Innovation
            VectorXd z_meas = Vector2d::Zero();
            z_meas << meas.range, meas.theta;
            // std::cout << "z_meas size: " << z_meas.size() << std::endl;
            VectorXd nu = normaliseLidarMeasurement(z_meas - z_mean);
            //std::cout << "Innovation is " << nu << std::endl;

            // Calculate the covariance of the transformed Sigma Points

            // Calculate S-Matrix
            MatrixXd S = MatrixXd::Zero(n_z, n_z);
            for(unsigned int i = 0; i < z_hat_mean_transformed_sigma_points.size(); ++i)
            {
                // difference between predicted measurement and mean
                VectorXd diff = normaliseLidarMeasurement(z_hat_mean_transformed_sigma_points[i] - z_mean);
                // std::cout << "Difference is " << diff << std::endl;
                S += weights[i] * diff * diff.transpose();
            }

            // std::cout << "S Matrix is " << S << std::endl;

            // Calculate Cross-Correlation Matrix
            MatrixXd cross_covariance_Pxz = MatrixXd::Zero(n_model_size, z_mean.size());
            for(unsigned int i = 0; i < sigma_points.size(); ++i)
            {
                //std::cout << "Sigma Point Size: " << sigma_points[i].size() << std::endl;
                //std::cout << "State Size: " << x_aug.size() << std::endl;
                //std::cout << "z_hat_mean_transformed_sigma_points Size: " << z_hat_mean_transformed_sigma_points[i].size() << std::endl;
                //std::cout << "z_mean Size: " << z_mean.size() << std::endl;
                VectorXd diff_state_x = normaliseState(sigma_points[i].head(n_model_size) - state);
                //std::cout << "Diff State is " << diff_state_x << std::endl;
                VectorXd diff_meas_z = normaliseLidarMeasurement(z_hat_mean_transformed_sigma_points[i] - z_mean);
                //std::cout << "Diff Meas is " << diff_meas_z << std::endl;
                cross_covariance_Pxz += weights[i] * diff_state_x * diff_meas_z.transpose();
                //std::cout << "Cross Covariance is " << cross_covariance_Pxz << std::endl;
            }

            // Calculate Kalman Gain
            MatrixXd K = cross_covariance_Pxz * S.inverse();
            // std::cout << "Kalman Gain is " << K << std::endl;
            // Update State and Covariance
            state = state + K * nu;
            state = normaliseState(state);
            cov = cov - K * S * K.transpose();
            // std::cout << "Updated State is " << state << std::endl;
            // std::cout << "Updated Covariance is " << cov << std::endl;

            setState(state);
            setCovariance(cov);
            
        }
        // ----------------------------------------------------------------------- //

    }
    else
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        //extract variables
        double px = state(0);
        double py = state(1);

         BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1) // Check that we have a valid beacon match
        {
            // calculate range and bearing
            double dx = map_beacon.x - px;
            double dy = map_beacon.y - py;

            // calculate components of measurement model
            double theta_hat;
            theta_hat = atan2(dy, dx);   

            // initialise psi state using lidar measurement
            state(2) = wrapAngle(theta_hat - meas.theta);

            // initalise velocity

            setState(state);
            setCovariance(cov);

            if(activate_non_zero_initial_condition and not fully_initialised(initialization_vector(state)))
            {
                reset();
            }


        }
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
        // HINT: Use the normaliseState() function to always keep angle values within correct range.
        // HINT: Do NOT normalise during sigma point calculation!
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        
        // Augment State

        // Generate Q Matrix
        MatrixXd Q = Matrix2d::Zero();
        Q(0,0) = GYRO_STD*GYRO_STD;
        Q(1,1) = ACCEL_STD*ACCEL_STD;

        // Augment the State Vector with Noise States
        int n_x = state.size();
        int n_w = 2;
        int n_aug = n_x + n_w;
        VectorXd x_aug = VectorXd::Zero(n_aug);
        MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
        x_aug.head(n_x) = state;
        P_aug.topLeftCorner(n_x,n_x) = cov;
        P_aug.bottomRightCorner(n_w,n_w) = Q;

        // Generate Sigma Points and Weights

        std::vector<VectorXd> sigma_points = generateSigmaPoints(x_aug, P_aug);
        std::vector<double> weights = generateSigmaWeights(n_aug);

        // Predict Sigma Points through Process Model
        std::vector<VectorXd> sigma_points_predict;
        for (const auto& sigma_point : sigma_points)
        {
            sigma_points_predict.push_back(vehicleProcessModel(sigma_point, gyro.psi_dot, dt));
        }

        // Predict State Mean

        state = VectorXd::Zero(n_x);
        
        for(unsigned int i = 0; i < sigma_points_predict.size(); ++i){
            state += weights[i] * sigma_points_predict[i];
        }
        state = normaliseState(state);

        // Predict State Covariance
        cov = MatrixXd::Zero(n_x,n_x);
        for(unsigned int i = 0; i < sigma_points_predict.size(); ++i){
            VectorXd diff = normaliseState(sigma_points_predict[i] - state);
            cov += weights[i] * diff * diff.transpose();
        }

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the UKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();
        int n_x = state.size();

        VectorXd z = Vector2d::Zero();
        int n_z = z.size();
        MatrixXd H = MatrixXd::Zero(n_z, n_x);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H(0,0) = 1;
        H(1,1) = 1;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat_mean_transformed_sigma_points = H * state;
        VectorXd y = z - z_hat_mean_transformed_sigma_points;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (MatrixXd::Identity(n_x, n_x) - K*H) * cov;

        if (innovation_in_bound(y, S) == true)
        {
            setState(state);
            setCovariance(cov);
        }
    }
    else if (not fully_initialised(initialization_vector(getState())))
    {
        reset();
        // calculate initial speed
        VectorXd state = getState();
        double dx = (meas.x - state(0));
        double dy = (meas.y - state(1));
        double time = 1;
        double v_x = dx/time;
        double v_y = dy/time;
        double v_initial = std::sqrt(v_x*v_x + v_y*v_y); // assume 1 second to reach GPS position
        if ((dx < 0) && (dy < 0))
        {
            v_initial *= -1;
        }
        
        //calculate initial heading
        double initial_heading = wrapAngle(atan2(dy, dx));

        state(0) = meas.x;
        state(1) = meas.y;
        state(2) = initial_heading;
        state(3) = v_initial;
        setState(state);
    }
    
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        int n_states = 4;

        if(activate_sensor_bias_correction == true)
        {
            n_states = 5;
        }

        VectorXd state = VectorXd::Zero(n_states);
        MatrixXd cov = MatrixXd::Zero(n_states,n_states);

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;
        
        if(activate_sensor_bias_correction == true)
        {
            cov(4,4) = bias_std*bias_std;
        }

        setState(state);
        setCovariance(cov);

        if(activate_non_zero_initial_condition and not fully_initialised(initialization_vector(state)))
        {
            reset();
        }
    }             
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
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
