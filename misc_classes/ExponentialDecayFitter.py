from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


class ExponentialDecayFitter(object):

    def __init__(self, x_data, y_data, model_type='exp_decay'):
        self.x_ = x_data
        self.y_ = y_data
        self.fit_model = model_type
        self.params = self.set_params()
        self.result = minimize(self.fit_fun, self.params, args=(self.x_, self.y_))
        self.eol_fit = self.find_eol_fit()

    def fit_fun(self, params, x, y=np.array([])):
        if self.fit_model == 'exp_decay':
            q0 = params['q0'].value
            tau = params['tau'].value
            beta = params['beta'].value
            model = self.exp_decay_fun(x, q0, tau, beta)
        elif self.fit_model == 'linear':
            k = params['k'].value
            model = self.linear_decay_fun(x, k)
        if y.size == 0:
            return model
        return y - model

    def plot_fit(self, ax):
        if not self.result:
            self.result = minimize(self.fit_fun, self.params, args=(self.x_, self.y_))
        ax.scatter(self.x_, self.y_, label='Raw data', color='orange')
        x = np.linspace(self.x_.min(), self.x_.max(), 250)
        ax.plot(x, self.fit_fun(self.result.params, x), label='Fit data', linestyle='dashed')
        return ax

    def find_eol_fit(self):
        params = [k.value for k in self.result.params.values()]
        return fsolve(self.solve_eol_fun, 200, args=(params))

    def set_params(self):
        params = Parameters()
        if self.fit_model == 'exp_decay':
            params.add('q0', value=1)
            params.add('tau', value=500)
            params.add('beta', value=1.4)
        elif self.fit_model == 'linear':
            params.add('k', value=2e-3)
        elif self.fit_model == 'linear_free':
            params.add('q0', value=1)
            params.add('k', value=2e-3)
        return params

    def solve_eol_fun(self, t, args, eol=0.7):
        q0 = args[0]
        tau = args[1]
        beta = args[2]
        return self.exp_decay_fun(t, q0, tau, beta) - eol

    @staticmethod
    def exp_decay_fun(t, q0, tau, beta):
        return q0 * np.exp(-(t / tau) ** beta)

    @staticmethod
    def linear_decay_fun(t, k):
        return 1 - k*t

    @staticmethod
    def linear_decay_fun_free(t, q0, k):
        return q0 - k * t


if __name__ == '__main__':
    x_f = np.tile(np.arange(0, 7*40, 40), 8)
    y_f = np.asarray([1., 0.96963788, 0.8829409, 0.79001824, 0.71034861, 0.63340969, 0.55585865, 1., 0.97381934,
                      0.89322034, 0.80722714, 0.73574004, 0.67201465, 0.59549534, 1.,
                      0.98448066, 0.92455064, 0.84844493, 0.7752184, 0.71429942, 0.65380638, 1., 0.98214782, 0.91183283,
                      0.8311451, 0.75486465, 0.68558798, 0.61261674, 1, 0.98395229, 0.92210435, 0.85039147, 0.77441516,
                      0.71267458, 0.64889158, 1, 0.98029529, 0.91113562, 0.83931963, 0.77473526, 0.70190701, 0.61777201,
                      1, 0.98319046, 0.91515155, 0.83250753, 0.75865168, 0.69045706, 0.61984013, 1, 0.98313809,
                      0.91042729, 0.81772053, 0.73465678, 0.66555213, 0.58471738
                      ])
    x_ = np.arange(0, 7*40, 40)
    y_ = np.array([[1, 0.96963788, 0.8829409, 0.79001824, 0.71034861, 0.63340969, 0.55585865],
                   [1, 0.97381934, 0.89322034, 0.80722714, 0.73574004, 0.67201465, 0.59549534],
                   [1., 0.98448066, 0.92455064, 0.84844493, 0.7752184, 0.71429942, 0.65380638],
                   [1., 0.98214782, 0.91183283, 0.8311451, 0.75486465, 0.68558798, 0.61261674],
                   [1, 0.98395229, 0.92210435, 0.85039147, 0.77441516, 0.71267458, 0.64889158],
                   [1., 0.98029529, 0.91113562, 0.83931963, 0.77473526, 0.70190701, 0.61777201],
                   [1, 0.98319046, 0.91515155, 0.83250753, 0.75865168, 0.69045706, 0.61984013],
                   [1., 0.98313809, 0.91042729, 0.81772053, 0.73465678, 0.66555213, 0.58471738]])
    t_eol = []
    fig, ax = plt.subplots(1, 1)
    for i in range(8):
        test_case = ExponentialDecayFitter(x_, y_[i, :].T)
        t_eol.append(test_case.find_eol_fit()[0])
        ax = test_case.plot_fit(ax)
        print(f'EOL for case {i} is {test_case.find_eol_fit()[0]:.2f} FCE')
    test_full = ExponentialDecayFitter(x_f, y_f)
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ax = test_case.plot_fit(ax)
    # ax = test_full.plot_fit(ax)
    ax.set_xlabel('FCE [-]', fontsize=15)
    ax.set_ylabel('Normalised capacity [-]', fontsize=15)
    param_vals = {k: test_case.result.params[k].value for k in test_case.result.params.keys()}
    param_str = '\n'.join([fr'{k}: {val:.2f}' for k, val in param_vals.items()])
    # param_str = '\n'.join(str(param_vals).strip('{').strip('}').split(','))
    plt.text(160, 0.9, param_str, fontsize=12, bbox=dict(facecolor='wheat', alpha=0.5))
