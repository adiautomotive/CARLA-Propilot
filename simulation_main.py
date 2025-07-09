def main(self, gui_options=None):
    if gui_options is None:
        # Provide default or fallback values here
        gui_options = {
            'vehicle_model': 'vehicle.tesla.model3',
            'lead_vehicle_model': 'vehicle.toyota.prius',
            'weather': 'ClearNoon'
        }

    return {
        'player_filter': gui_options['vehicle_model'],
        'lead_filter': gui_options['lead_vehicle_model'],
        'weather': gui_options['weather']
    }
