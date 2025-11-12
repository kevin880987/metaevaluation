from pymle.models import GeometricBM
from pymle.sim.Simulator1D import Simulator1D
from pymle.core.TransitionDensity import ExactDensity, KesslerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np
from pymle.core.TransitionDensity import KesslerDensity, ShojiOzakiDensity
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def read_HI_data(path):
    UA = pd.read_csv(path,index_col=0)
    A = UA.values

    return A

def parameter_estimation(sample,dt):
    model = GeometricBM()
    param_bounds = [(0, 1), (0, 1)]
    guess = np.array([0.1, 0.4])
    kessler_est = AnalyticalMLE(sample, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess)
    mu = kessler_est.params[0]
    sigma = kessler_est.params[1]
    return mu, sigma

def get_simulated_process(original_HI,S0,mu,sigma,dt):
    
    def geometric_brownian_motion(S0, mu, sigma, T, dt, paths):
        """
        Simulates geometric Brownian motion using Euler's method.
        
        Args:
            S0 (float): Initial stock price.
            mu (float): Drift (expected return) of the stock.
            sigma (float): Volatility (standard deviation) of the stock.
            T (float): Time period for simulation.
            dt (float): Time step for simulation.
            paths (int): Number of simulation paths to generate.
        
        Returns:
            numpy.ndarray: Array of simulated stock prices.
        """
        N = int(T / dt)  # Number of time steps
        dW = np.sqrt(dt) * np.random.randn(paths, N)  # Wiener process increment
        
        # Initialize stock price array with initial value
        S = np.zeros((paths, N + 1))
        S[:, 0] = S0
        
        for i in range(1, N + 1):
            # Update stock price using Euler's method
            dS = mu * S[:, i - 1] * dt + sigma * S[:, i - 1] * dW[:, i - 1]
            S[:, i] = S[:, i - 1] + dS
        
        return S
    
    def generate_simulated_path(S0,mu,sigma,dt):
        # Parameters for simulation
        S0 = S0 # Initial stock price
        mu = mu # Drift (expected return) of the stock
        sigma = sigma# Volatility (standard deviation) of the stock
        T = 2092  # Time period for simulation
        paths = 2000  # Number of simulation paths to generate

        simulated_paths = geometric_brownian_motion(S0, mu, sigma, T, dt, paths)

        return simulated_paths
    
    def plot_simualted_result(original_data,simulated_paths):
        plt.figure(figsize=(21,9))
        for i in range(0,10):
            plt.plot(simulated_paths[i])
        plt.plot(original_data,color="green",linewidth=5,marker=".")
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Health Index",fontsize=15)

    def plot_confidence_interval(original_data,simulated_paths):
        j = simulated_paths

        ci10=[]
        ci90=[]
        for i in range(j.shape[1]):
            a = j[:,i]
            ci10.append(np.percentile(a,10))
            ci90.append(np.percentile(a,90))
        
        plt.figure(figsize=(21,9))
        plt.plot(original_data)
        plt.plot(ci10)
        plt.plot(ci90)

    sp = generate_simulated_path(S0,mu,sigma,dt)

    plot_simualted_result(original_HI,sp)

    plot_confidence_interval(original_HI,sp)

    return sp

def process_rul(original_data,alarm_threshold_point,failure_point,simulated_paths):

    sample = original_data
    alarm_threshold = sample.T[0][alarm_threshold_point]
    failure = sample.T[0][failure_point]

    j = simulated_paths

    rul=[]
    for a in j:
        try:
            a = list(a)
            res = list(filter(lambda i: i > alarm_threshold, a))[0]
            T1 = a.index(res)
            try:
                res2 = list(filter(lambda i: i > failure , a))[0]
                T2 = a.index(res2)
            except:
                T2 = len(a)
            rul.append(T2-T1)
        except:
            continue

    plt.figure(figsize=(21,9))
    sns.histplot(rul,color="red")
    plt.xticks(fontsize="20")
    plt.yticks(fontsize="20")
    plt.xlabel("Remaining Useful Life", fontsize="20")
    plt.ylabel("Count",fontsize="20")


    real_rul = failure_point - alarm_threshold_point
    ex_rul = np.mean(rul)
    ex_std = np.std(rul)
    print(f"Real_RUL = {failure_point}-{alarm_threshold_point}={real_rul}")
    print(f"Experiment_RUL = {ex_rul}")
    print(f"Experiment_RUL_STD = {ex_std}")

    return ex_rul

def real_option_analysis(original_data,sp,C_prix,C_dm,C_rep,I_penalty,date_range,t0,failure,delta_i):

    C_prix = C_prix
    C_dm = C_dm
    C_rep = C_rep
    I_penalty = I_penalty

    candidate_date = range(1,date_range)

    def calculate_number_of_failure(j,tk):
        failure = original_data[1900][0]
        number_of_failure = 0
        for i in j[:,:tk]:
            if sum(i > failure) > 0 :
                number_of_failure += 1
        
        return number_of_failure
    
    cost_of_each_status={}
    for i in range(1,11):
        cost_of_each_status[i]=6**(i-3)

    dic={}

    t0= t0
    print(original_data[1900])
    failure = original_data[1900][0]

    for i in candidate_date[:9]:

        tk = t0 + delta_i*i
        
        dic[i] = {}

        if calculate_number_of_failure(sp,tk)==0:
            
            p_tf_st_tk = 10*i/2000
        
        else:
            
            p_tf_st_tk = calculate_number_of_failure(sp,tk)/2000

        p_tf_bt_tk = 1 - p_tf_st_tk


        dic[i]["ER"] = p_tf_bt_tk*C_prix*1000*i

        # C_pr_sup = return_prventive_cost_of_a_machine(np.mean(j[j[:,tk] < failure][:,tk]))
        # C_pr_sup = C_dm*(np.mean(sp[sp[:,tk] < failure][:,tk])-np.mean(sp[sp[:,tk] < failure][:,t0]))
        C_pr_sup = cost_of_each_status[i+1]

        
        lis_temp = sp[sp[:,tk] > failure]


        penalty_length =[]
        
        for a in lis_temp:
            try:
                a = list(a)
                res = list(filter(lambda i: i > failure, a[int(t0):int(tk)]))[0]
                T1 = a.index(res)
                penalty_length.append(tk-T1)
            except:
                continue
        if len(penalty_length) == 0:
            C_cr_sup = 0
        else:
            C_cr_sup = C_rep + I_penalty*(np.nanmean(penalty_length))*1000
        # C_cr_sup = C_rep + I_penalty*(np.nanmean(penalty_length))*1000

        dic[i]["p_tf_bt_tk"]= p_tf_bt_tk
        dic[i]["p_tf_st_tk"]= p_tf_st_tk
        
        dic[i]["C_pr_sup"] = C_pr_sup
        dic[i]["C_cr_sup"] = C_cr_sup


        dic[i]["EC"] = p_tf_bt_tk*C_pr_sup + p_tf_st_tk*C_cr_sup


        dic[i]["Total Return"] = dic[i]["ER"]-dic[i]["EC"]
    
    a =pd.DataFrame(dic).T

    plt.figure(figsize=(21,9))
    for i in ["ER","EC","Total Return"]:
        plt.plot(a.loc[:8,i],label=i)
    plt.legend(fontsize="15")
    plt.ylabel("NTD",fontsize="15")
    plt.xlabel("Time",fontsize="15")
    plt.xticks(fontsize="15")
    plt.yticks(fontsize="15")

    bt = a.idxmax()["Total Return"]

    print(f"{bt} is the best time to maintain!")

def main():

    path = input()

    A = read_HI_data(path)

    mu,sigma = parameter_estimation(A,1)

    sp = get_simulated_process(A,1,mu,sigma,1)

    rul = process_rul(A,1472,1750,sp)

    real_option_analysis(A,sp,5.5,2000,3000,2,10,1450,1750,30)

main()


