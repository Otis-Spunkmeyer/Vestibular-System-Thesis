import numpy as np

class ComplementaryFilter:
    """
    Estimates body-in-space angle from two diametrically-opposed IMUs.

    Each IMU provides:
        gyro - angular velocity in rad/s
        acc - accelometer vector [ax, ay, az] in the sensor frame, g

    Diametrically-opposed mounting means two IMU's are 180 degrees apart
    on the same rigid body segement. Averaging their acceleration reduces
    linear acceleration artifacts for a purer gravity estimate for tilt.
    
    alpha: high-pass weight on gyroscope integration (default = 0.98)
        Higher alpha: more gyro-dominate (better for fast motion)
        Lower alpha:  more accel-dominate (better for slow drift correction).
    """
    def __init__(self):
        alpha = float(input("Enter complementary filter alpha (0–1, recommended 0.98): "))
        if alpha > 1 or alpha < 0:
            raise ValueError('Alpha must be a value between 0 and 1.')
        self.alpha = alpha
        self.angle = 0.0  # current body-in-space angle estimate (rad)

    def update(self, gyro1, gyro2, acc1, acc2, dt):
        '''
        Parameters
        ----------
        gyro1, gyro2: float - Y-axis angular velocity from each IMU (rad/s)
        acc1, acc2  : array - [ax, az] accelerometer readings from each IMU (g)
        dt          : float - timestep in seconds

        Returns
        -------
        float - estimated body in space angle (rad)
        '''
        # Average the gyro - both measure the same rotation, averaging reduces the noise
        gyro_mean = (gyro1 + gyro2) / 2

        # average accelerometers - cancels common-mode linear acceleration
        # (works because the two IMU's are diameterically opposed on same rigid body)
        acc_mean = (np.array(acc1) +np.array(acc2)) / 2

        # Tilt angle from gravity direction (valid only during static/slow motion)
        accel_angle = np.arctan2(acc_mean[0], acc_mean[1])

        # Complementary filter: combines gyro and accel readings with weights to create angle
        self.angle = self.alpha*(self.angle +gyro_mean*dt) + (1.0-self.alpha) * accel_angle

        return self.angle 
    
    def reset(self, angle=0.0):
        self.angle = angle