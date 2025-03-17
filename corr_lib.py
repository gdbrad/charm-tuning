import lsqfit
import numpy as np
import gvar as gv
import warnings
import matplotlib.pyplot as plt

class Fitter(object):

    def __init__(self, n_states, prior, t_start, t_end,
                correlators, particle_statistics, t_period=None):

        if t_period is None:
            # Basically, this gets the length of the correlators gvar objects
            # Since the ps and ss should have the same length, we just use whichever
            # key is first in the dictionary
            t_period  = len(correlators[list(correlators)[0]])

        if t_end is None:
            t_end = t_period - t_start

        # construct effective mass, wf
        effective_mass = {}
        effective_wf = {}
        for snk in correlators:
            t = np.arange(len(correlators[snk]))
            if particle_statistics == 'fermi-dirac':
                effective_mass[snk] = np.log(correlators[snk] / np.roll(correlators[snk] , -1))
                effective_wf[snk] = np.exp(effective_mass[snk] *t) *correlators[snk]

            elif particle_statistics == 'bose-einstein':
                effective_mass[snk] = np.arccosh(
                    (np.roll(correlators[snk] , -1) + np.roll(correlators[snk] , 1))
                        /(2*correlators[snk] ))
                effective_wf[snk] = 1 / np.cosh(effective_mass[snk] *(t - t_period/2)) *correlators[snk]

        self.t_period = t_period
        self.n_states = n_states
        self.t_start = t_start
        self.t_end = t_end
        self.particle_statistics = particle_statistics
        self.prior = prior
        self.correlators = correlators
        self.effective_mass = effective_mass
        self.effective_wf = effective_wf

        self.fits = {}
        

    def fcn_effective_mass(self, t, t_start=None, t_end=None, n_states=None):
        if t_start is None: t_start = self.t_start
        if t_end is None: t_end = self.t_end
        if n_states is None: n_states = self.n_states

        p = self.get_fit(t_start, t_end, n_states).p
        output = {}
        for model in self._make_models_simult_fit(t_start, t_end, n_states):
            snk = model.datatag
            output[snk] = model.fcn_effective_mass(p, t)
        return output


    def fcn_effective_wf(self, t, t_start=None, t_end=None, n_states=None):
        if t_start is None: t_start = self.t_start
        if t_end is None: t_end = self.t_end
        if n_states is None: n_states = self.n_states

        p = self.get_fit(t_start, t_end, n_states).p
        output = {}
        for model in self._make_models_simult_fit(t_start, t_end, n_states):
            snk = model.datatag
            output[snk] = model.fcn_effective_wf(p, t)
        return output


    def get_fit(self, t_start=None, t_end=None, n_states=None):
        if t_start is None: t_start = self.t_start
        if t_end is None: t_end = self.t_end
        if n_states is None: n_states = self.n_states

        # Don't rerun the fit if it's already been made
        if (t_start, t_end, n_states) not in self.fits:
            self.fits[(t_start, t_end, n_states)] = self._make_fit(t_start, t_end, n_states)
        return self.fits[(t_start, t_end, n_states)]


    def get_energies(self, t_start=None, t_end=None, n_states=None):
        if t_start is None: t_start = self.t_start
        if t_end is None: t_end = self.t_end
        if n_states is None: n_states = self.n_states
        
        temp_fit = self.get_fit(t_start, t_end, n_states)

        output = gv.gvar(np.zeros(self.n_states))
        output[0] = temp_fit.p['E0']
        for k in range(1, self.n_states):
            output[k] = output[0] + np.sum([(temp_fit.p['dE'][j]) for j in range(k)], axis=0)
        return output


    def _make_fit(self, t_start, t_end, n_states):
        # Make fit with lsqfit.MultiFitter
        # Essentially: first we create a model (which is a subclass of MultiFitter)
        # Then we make a fitter using the models
        # Finally, we make the fit with our two sets of correlators

        models = self._make_models_simult_fit(t_start, t_end, n_states)

        fitter = lsqfit.MultiFitter(models=models)
        fit = fitter.lsqfit(data=self.correlators,
                            prior=self._make_prior(self.prior, n_states))
        self.fit = fit
        return fit


    def _make_models_simult_fit(self, t_start, t_end, n_states):
        models = np.array([])

        if self.particle_statistics == 'fermi-dirac':
            for key in list(self.correlators):
                param_keys = {
                    'E0'      : 'E0',
                    'log(dE)' : 'log(dE)',
                    'wf'      : 'wf_'+key,
                }
                models = np.append(models,
                           BaryonModel(key, t=range(t_start, t_end),
                           param_keys=param_keys, n_states=n_states))

        if self.particle_statistics == 'bose-einstein':
            for key in list(self.correlators):
                param_keys = {
                    'log(E0)' : 'log(E0)',
                    'log(dE)' : 'log(dE)',
                    'wf'      : 'wf_'+key,
                }
                models = np.append(models,
                           MesonModel(key, t=range(t_start, t_end), t_period=self.t_period,
                           param_keys=param_keys, n_states=n_states))
        return models


    # Converts normally-distributed energy priors to log-normally-distributed priors,
    # thereby forcing each excited state to be positive
    # and larger than the previous excited state
    def _make_prior(self, prior, n_states):
        resized_prior = {}
        for key in list(prior):
            resized_prior[key] = prior[key][:self.n_states]
        new_prior = resized_prior.copy()

        if self.particle_statistics=='bose-einstein':
            new_prior['log(E0)'] = np.log(resized_prior['E'][0])

        elif self.particle_statistics=='fermi-dirac':
            new_prior['E0'] = resized_prior['E'][0]

        # Don't need this entry
        new_prior.pop('E', None)

        # We force the energy to be positive by using the log-normal dist of dE
        # let log(dE) ~ eta; then dE ~ e^eta
        new_prior['log(dE)'] = gv.gvar(np.zeros(len(resized_prior['E']) - 1))
        for j in range(len(new_prior['log(dE)'])):
            temp_gvar = gv.gvar(gv.mean(resized_prior['E'][j+1] - resized_prior['E'][j]),
                                gv.sdev(resized_prior['E'][j+1]))

            new_prior['log(dE)'][j] = np.log(temp_gvar)

        return new_prior


    def _plot_quantity(self, quantity,
            tmin, tmax, ylabel=None, ylim=None):

        fig, ax = plt.subplots()
        
        colors = ['rebeccapurple', 'mediumseagreen']
        for j, snk in enumerate(sorted(quantity)):
            x = np.arange(tmin, tmax)
            y = gv.mean(quantity[snk])[x]
            y_err = gv.sdev(quantity[snk])[x]

            ax.errorbar(x, y, xerr = 0.0, yerr=y_err, fmt='o', capsize=5.0,
                        color=colors[j%len(quantity)], capthick=2.0, alpha=0.6, elinewidth=5.0, label=snk)

        # Label dirac/smeared data
        #plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.legend(loc=3, bbox_to_anchor=(0,1), ncol=len(quantity))
        plt.grid(True)
        plt.xlabel('$t$', fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        if ylim is not None:
            plt.ylim(ylim)
        fig = plt.gcf()
        return fig


    def plot_effective_mass(self, tmin=None, tmax=None, ylim=None, show_fit=True):
        if tmin is None: tmin = 1
        if tmax is None: tmax = self.t_period - 1

        fig = self._plot_quantity(
            quantity=self.effective_mass, 
            ylabel=r'$m_\mathrm{eff}$', 
            tmin=tmin, tmax=tmax, ylim=ylim) 

        if show_fit:
            ax = plt.gca()

            colors = ['rebeccapurple', 'mediumseagreen']
            t = np.linspace(tmin, tmax)
            effective_mass_fit = self.fcn_effective_mass(t=t)
            for j, snk in enumerate(sorted(effective_mass_fit)):
                color = colors[j%len(colors)]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                ax.plot(t, pm(effective_mass_fit[snk], 0), '--', color=color)
                ax.plot(t, pm(effective_mass_fit[snk], 1), 
                            t, pm(effective_mass_fit[snk], -1), color=color)
                ax.fill_between(t, pm(effective_mass_fit[snk], -1), pm(effective_mass_fit[snk], 1), facecolor=color, alpha = 0.10, rasterized=True)

        fig = plt.gcf()
        # plt.close()
        return fig


    def plot_effective_wf(self, tmin=None, tmax=None, ylim=None, show_fit=True):
        if tmin is None: tmin = 1
        if tmax is None: tmax = self.t_period - 1

        fig = self._plot_quantity(
            quantity=self.effective_wf, 
            ylabel=r'$A_\mathrm{eff}$',
            tmin=tmin, tmax=tmax, ylim=ylim)

        if show_fit:
            ax = plt.gca()

            colors = ['rebeccapurple', 'mediumseagreen']
            t = np.linspace(tmin, tmax)
            effective_wf_fit = self.fcn_effective_wf(t=t)
            for j, snk in enumerate(sorted(effective_wf_fit)):
                color = colors[j%len(colors)]

                pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
                ax.plot(t, pm(effective_wf_fit[snk], 0), '--', color=color)
                ax.plot(t, pm(effective_wf_fit[snk], 1), 
                            t, pm(effective_wf_fit[snk], -1), color=color)
                ax.fill_between(t, pm(effective_wf_fit[snk], -1), pm(effective_wf_fit[snk], 1), facecolor=color, alpha = 0.10, rasterized=True)

        fig = plt.gcf()
        plt.close()
        return fig


# This class is needed to instantiate an object for lsqfit.MultiFitter
# Used for particles that obey fermi-dirac statistics
class BaryonModel(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, param_keys, n_states):
        super(BaryonModel, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.n_states = n_states

        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys


    def fitfcn(self, p, t=None):
        if t is None:
            t = self.t

        wf = p[self.param_keys['wf']]
        E0 = p[self.param_keys['E0']]
        dE = np.exp(p[self.param_keys['log(dE)']])

        output = wf[0] * np.exp(-E0 * t)
        for j in range(1, self.n_states):
            E_j = E0 + np.sum([dE[k] for k in range(j)], axis=0)
            output = output + wf[j] * np.exp(-E_j * 96-t)

        return output


    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        # Extract the model's parameters from prior.
        return prior


    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag][self.t]


    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.log(self.fitfcn(p, t) / self.fitfcn(p, t+1))


    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        t = np.array(t)
        
        return np.exp(self.fcn_effective_mass(p, t)*t) * self.fitfcn(p, t)


# Used for particles that obey bose-einstein statistics
class MesonModel(lsqfit.MultiFitterModel):
    def __init__(self, datatag, t, t_period, param_keys, n_states):
        super(MesonModel, self).__init__(datatag)
        # variables for fit
        self.t = np.array(t)
        self.t_period = t_period
        self.n_states = n_states

        # keys (strings) used to find the wf_overlap and energy in a parameter dictionary
        self.param_keys = param_keys


    def fitfcn(self, p, t=None):

        if t is None:
            t = self.t

        wf = p[self.param_keys['wf']]
        E0 = np.exp(p[self.param_keys['log(E0)']])
        dE = np.exp(p[self.param_keys['log(dE)']])

        output = wf[0] * np.cosh( E0 * (t - self.t_period/2.0) )
        for j in range(1, self.n_states):
            E_j = E0 + np.sum([dE[k] for k in range(j)], axis=0)
            output = output + wf[j] * np.cosh( E_j * (t - self.t_period/2.0) )

        return output


    # The prior determines the variables that will be fit by multifitter --
    # each entry in the prior returned by this function will be fitted
    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        # Extract the model's fit data from data.
        # Key of data must match model's datatag!
        return data[self.datatag][self.t]


    def fcn_effective_mass(self, p, t=None):
        if t is None:
            t=self.t

        return np.arccosh((self.fitfcn(p, t-1) + self.fitfcn(p, t+1))/(2*self.fitfcn(p, t)))


    def fcn_effective_wf(self, p, t=None):
        if t is None:
            t=self.t
        return 1 / np.cosh(self.fcn_effective_mass(p, t)*(t - self.t_period/2)) * self.fitfcn(p, t)
