import hapticAPI

hapticAPI.initiate_config()
hapticAPI.show_modules()

# activate_motor(module_id,intensity,duration) Intensity = 0 to 100,
# Duration in milliseconds
# hapticAPI.delay(Duration in milliseconds)


hapticAPI.activate_motor(1, 100, 4000)
hapticAPI.activate_motor(3, 60, 7000)
hapticAPI.activate_motor(4, 30, 3000)

# continuous_motion(intensity, duration, delay_time)
hapticAPI.continuous_motion(80, 500, 10)