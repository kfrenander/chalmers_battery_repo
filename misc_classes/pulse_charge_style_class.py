class TestCaseStyler:
    def __init__(self):
        # Define base styles (color, marker, linestyle) once
        self.base_styles = {
            '10 Hz 25 duty cycle pulse': {'color': '#0071bc', 'marker': 'o', 'linestyle': '--'},
            '10 Hz 50 duty cycle pulse': {'color': '#d85218', 'marker': 's', 'linestyle': '--'},
            '125 Hz 25 duty cycle pulse': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
            '1C reference': {'color': '#d62728', 'marker': 'x', 'linestyle': '-'},
            '1 Hz 25 duty cycle pulse': {'color': '#9467bd', 'marker': '.', 'linestyle': ':'},
            '1 Hz 50 duty cycle pulse': {'color': '#8c564b', 'marker': '8', 'linestyle': ':'},
            '50 Hz 25 duty cycle pulse': {'color': '#e377c2', 'marker': 'v', 'linestyle': '-.'},
            '50 Hz 50 duty cycle pulse': {'color': '#7f7f7f', 'marker': '+', 'linestyle': '-.'},
            '1000mHz Pulse Charge': {'color': '#004488', 'marker': 'x', 'linestyle': '--'},
            '1000mHz Pulse Charge no pulse discharge': {'color': '#22b2b2', 'marker': 'H', 'linestyle': ':'},
            '100mHz Pulse Charge': {'color': '#76ab2f', 'marker': '+', 'linestyle': '-.'},
            '100mHz Pulse Charge no pulse discharge': {'color': '#FF6347', 'marker': '^', 'linestyle': '--'},
            '500mHz Pulse Charge': {'color': '#32CD32', 'marker': 'h', 'linestyle': ':'},
            '10mHz Pulse Charge': {'color': '#FFD700', 'marker': '*', 'linestyle': '-'},
            'Reference test constant current': {'color': '#3f3f3f', 'marker': 'o', 'linestyle': '--'},
            '320mHz Pulse Charge': {'color': '#6e6a6a', 'marker': 'H', 'linestyle': '-.'},
        }
        # self.base_styles = {
        #     '10 Hz 25 duty cycle pulse': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '--'},
        #     '10 Hz 50 duty cycle pulse': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
        #     '125 Hz 25 duty cycle pulse': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
        #     '1C reference': {'color': '#d62728', 'marker': 'x', 'linestyle': '-'},
        #     '1 Hz 25 duty cycle pulse': {'color': '#9467bd', 'marker': '.', 'linestyle': ':'},
        #     '1 Hz 50 duty cycle pulse': {'color': '#8c564b', 'marker': '8', 'linestyle': ':'},
        #     '50 Hz 25 duty cycle pulse': {'color': '#e377c2', 'marker': 'v', 'linestyle': '-.'},
        #     '50 Hz 50 duty cycle pulse': {'color': '#7f7f7f', 'marker': '+', 'linestyle': '-.'},
        #     '1000mHz Pulse Charge': {'color': '#800000', 'marker': 'p', 'linestyle': '--'},
        #     '1000mHz Pulse Charge no pulse discharge': {'color': '#A52A2A', 'marker': 'H', 'linestyle': ':'},
        #     '100mHz Pulse Charge': {'color': '#D73027', 'marker': 'D', 'linestyle': '-.'},
        #     '100mHz Pulse Charge no pulse discharge': {'color': '#FF6347', 'marker': '^', 'linestyle': '--'},
        #     '500mHz Pulse Charge': {'color': '#32CD32', 'marker': 'h', 'linestyle': ':'},
        #     '10mHz Pulse Charge': {'color': '#FFD700', 'marker': '*', 'linestyle': '-'},
        #     'Reference test constant current': {'color': '#000000', 'marker': 'o', 'linestyle': '--'},
        #     '320mHz Pulse Charge': {'color': '#6e6a6a', 'marker': 'H', 'linestyle': '-.'},
        # }

        # Define labels separately
        self.full_labels = {
            '10 Hz 25 duty cycle pulse': '10 Hz 25% Duty Pulse',
            '10 Hz 50 duty cycle pulse': '10 Hz 50% Duty Pulse',
            '125 Hz 25 duty cycle pulse': '125 Hz 25% Duty Pulse',
            '1C reference': '1C Reference',
            '1 Hz 25 duty cycle pulse': '1 Hz 25% Duty Pulse',
            '1 Hz 50 duty cycle pulse': '1 Hz 50% Duty Pulse',
            '50 Hz 25 duty cycle pulse': '50 Hz 25% Duty Pulse',
            '50 Hz 50 duty cycle pulse': '50 Hz 50% Duty Pulse',
            '1000mHz Pulse Charge': '1000 mHz Pulse Charge',
            '1000mHz Pulse Charge no pulse discharge': '1000 mHz No Discharge Pulse',
            '100mHz Pulse Charge': '100 mHz Pulse Charge',
            '100mHz Pulse Charge no pulse discharge': '100 mHz No Discharge Pulse',
            '500mHz Pulse Charge': '500 mHz Pulse Charge',
            '10mHz Pulse Charge': '10 mHz Pulse Charge',
            'Reference test constant current': 'Reference 2.5 A',
            '320mHz Pulse Charge': '320 mHz Pulse Charge',
        }

        self.abbreviated_labels = {
            '10 Hz 25 duty cycle pulse': '10 Hz PC-PD-25',
            '10 Hz 50 duty cycle pulse': '10 Hz PC-PD-50',
            '125 Hz 25 duty cycle pulse': '125 Hz PC-PD-25',
            '1C reference': '1C ref',
            '1 Hz 25 duty cycle pulse': '1 Hz PC-PD-25',
            '1 Hz 50 duty cycle pulse': '1 Hz PC-PD-50',
            '50 Hz 25 duty cycle pulse': '50 Hz PC-PD-25',
            '50 Hz 50 duty cycle pulse': '50 Hz PC-PD-50',
            '1000mHz Pulse Charge': '1000 mHz PC-PD',
            '1000mHz Pulse Charge no pulse discharge': '1000 mHz PC-NPD',
            '100mHz Pulse Charge': '100 mHz PC-PD',
            '100mHz Pulse Charge no pulse discharge': '100 mHz PC-NPD',
            '500mHz Pulse Charge': '500 mHz PC-PD',
            '10mHz Pulse Charge': '10 mHz PC-PD',
            'Reference test constant current': 'CC-ref',
            '320mHz Pulse Charge': '320 mHz PC-PD',
        }

    def get_style(self, case_name, abbreviated=False):
        """
        Returns the style dictionary for the given test case name.
        If the case is not found, returns a default style.
        """
        style = self.base_styles.get(case_name, {'color': 'gray', 'marker': '.', 'linestyle': '-'})
        label_dict = self.abbreviated_labels if abbreviated else self.full_labels
        return {**style, 'label': label_dict.get(case_name, case_name)}

    def get_abbrv_style(self, case_name):
        """
        Maintains compatibility with previous versions.
        Calls get_style with abbreviated=True.
        """
        return self.get_style(case_name, abbreviated=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('widthsixinches')

    styler = TestCaseStyler()
    x = np.linspace(0, 4*np.pi, 100)
    for i, key in enumerate(styler.base_styles.keys()):
        plt.plot(x, np.sin(x) + 2*i, **styler.get_abbrv_style(key))
    plt.legend()
