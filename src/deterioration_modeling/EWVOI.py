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
from discreteMarkovChain import markovChain


def read_HI_data(path):
    UA = pd.read_csv(path,index_col=0)
    A = UA.values
    print(A)
    print(A.T)
    return A

def show_variance_of_increments(A):

    plt.figure(figsize=(21,9))
    plt.plot(np.diff(np.log(A.T[0])))
    plt.xlabel("Time(hr)",fontsize=25)
    plt.ylabel("Log Increment",fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    ts = pd.Series(np.diff(np.log(A.T[0])))
    plt.figure(figsize=(21,9))
    ts.rolling(window=100).var().plot(style='b')
    plt.xlabel("Time(hr)",fontsize=25)
    plt.ylabel("Variance of Increments",fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)

    return ts

def parameter_estimation(sample,dt,ts):

    # Construct Transition Matrix and Get Steady State
    overall_variance = ts.rolling(window=2092).var().iloc[-1]

    a = ts.groupby(ts.index//5).var()

    state_series = a.copy()

    state_series[a<=overall_variance]=0

    state_series[(a>=overall_variance) & (a<=2*overall_variance)]=1

    state_series[a>=2*overall_variance]=2

    state_series=state_series.astype(int)

    t = list(state_series)

    def transition_matrix(t):
        n = 1+ max(t) #number of states

        M = [[0]*n for _ in range(n)]

        for (i,j) in zip(t,t[1:]):
            M[i][j] += 1

        #now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        return M
    
    m = transition_matrix(t)

    print(m)

    mc = markovChain(np.array(m))
    mc.computePi('linear')
    print("***************")
    print(mc.pi)

    # Estimate each state mu and sigma

    s1_lis=[]
    s2_lis=[]
    s3_lis=[]


    for i in range(len(t)):

        data = ts.groupby(ts.index//5).get_group(i).values
        
        if t[i]==0:
            for j in data:
                s1_lis.append(j)
        elif t[i]==1:
            for j in data:
                s2_lis.append(j)
        else :
            for j in data:
                s3_lis.append(j)
    
    def generate_time_series_to_estimate(start,s1):
        start = start

        s1_lis = s1
        
        ts = [start]

        for i in range(len(s1_lis)):

            next_item = ts[-1]*np.exp(s1_lis[i])

            ts.append(next_item)
        print(ts)
        print(len(ts))

        plt.figure(figsize=(21,9))
        plt.plot(ts)
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Health Index",fontsize=15)
        
        return np.reshape(ts,(len(ts),1))
    
    s1=generate_time_series_to_estimate(1,s1_lis)
    s2=generate_time_series_to_estimate(1,s2_lis)
    s3=generate_time_series_to_estimate(1,s3_lis)

    def parameter_estimation(sample):
        model = GeometricBM()
        dt = 1/(len(sample))
        param_bounds = [(0, 1), (0, 1)]
        guess = np.array([0.1, 0.4])
        kessler_est = AnalyticalMLE(sample, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess)
        mu = kessler_est.params[0]
        sigma = kessler_est.params[1]
        return mu, sigma
    
    s1_mu,s1_sig = parameter_estimation(s1)
    s2_mu,s2_sig = parameter_estimation(s2)
    s3_mu,s3_sig = parameter_estimation(s3)
    print(s1_mu,s2_mu,s3_mu)
    return s1_mu,s1_sig,s2_mu,s2_sig,s3_mu,s3_sig,mc.pi

def get_simulated_process(original_data,S0,T,dt,s1_m,s1_sig,s2_m,s2_sig,s3_m,s3_sig,steady_state,paths):

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
        print(S)
        return S
    
    def simualte_advance_model(s1_mu,s1_sig,s2_mu,s2_sig,s3_mu,s3_sig,steady_state,S0,T,dt,paths):

        S0 = S0
        T = T
        dt = 1/2092
        paths = paths
        s1_mu = s1_mu + 0.56
        s1_sig = 0.07

        a = geometric_brownian_motion(S0, s1_mu, s1_sig, T, dt, paths)
        b = geometric_brownian_motion(S0, s2_mu, s2_sig, T, dt, paths)
        c = geometric_brownian_motion(S0, s3_mu, s3_sig, T, dt, paths)

        result = 0.99*a + 0.01*b+ 0*c
        # result = steady_state[0]*a + steady_state[1]*b+ steady_state[2]*c

        return result
    
    r = simualte_advance_model(s1_m,s1_sig,s2_m,s2_sig,s3_m,s3_sig,steady_state,S0,T,dt,paths)

    def plot_simualted_result(original_data,simulated_paths):
        plt.figure(figsize=(21,9))
        for i in range(0,10):
            plt.plot(simulated_paths[i])
        plt.plot(original_data,color="green",linewidth=5,marker=".")
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Health Index",fontsize=15)

    plot_simualted_result(original_data,r)

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
    
    plot_confidence_interval(original_data,r)

    return r

def process_rul(original_data,alarm_threshold_point,failure_point,simulated_paths):

    sample = original_data
    alarm_threshold = sample.T[0][alarm_threshold_point]
    print(sample.T[0][alarm_threshold_point], "hiiii")
    failure = sample.T[0][failure_point]
    print(sample.T[0][failure_point], "hiiii")

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
            print(T2,T1)
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

def real_option_analysis(original_data,sp,C_prix,C_dm,C_rep,I_penalty,date_range,t0,failure,delta_i,paths):

    C_prix = C_prix
    C_dm = C_dm
    C_rep = C_rep
    I_penalty = I_penalty

    candidate_date = range(1,date_range)

    def calculate_number_of_failure(j,tk):
        failure = original_data[1760][0]
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

    failure = original_data[1760][0]

    for i in candidate_date[:11]:

        tk = t0 + delta_i*i
        
        dic[i] = {}

        if calculate_number_of_failure(sp,tk)==0:
            p_tf_st_tk = 10*i/paths
        
        else:
            
            p_tf_st_tk = calculate_number_of_failure(sp,tk)/paths

        p_tf_bt_tk = 1 - p_tf_st_tk


        dic[i]["ER"] = p_tf_bt_tk*C_prix*1000*i*delta_i
      
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

        C_cr_sup = C_rep + I_penalty*(np.nanmean(penalty_length))*1000

        dic[i]["p_tf_bt_tk"]= p_tf_bt_tk
        dic[i]["p_tf_st_tk"]= p_tf_st_tk
        
        dic[i]["C_pr_sup"] = C_pr_sup
        dic[i]["C_cr_sup"] = C_cr_sup


        dic[i]["EC"] = p_tf_bt_tk*C_pr_sup + p_tf_st_tk*C_cr_sup


        dic[i]["Total Return"] = dic[i]["ER"]-dic[i]["EC"]
    
    a =pd.DataFrame(dic).T

    print(a)

    plt.figure(figsize=(21,9))
    for i in ["ER","EC","Total Return"]:
        plt.plot(a.loc[:,i],label=i)
    plt.legend(fontsize="15")
    plt.ylabel("NTD",fontsize="15")
    plt.xlabel("Time",fontsize="15")
    plt.xticks(fontsize="15")
    plt.yticks(fontsize="15")

    bt = a.idxmax()["Total Return"]

    print(f"{bt} is the best time to maintain!")

def main():

    # path = input()

    # A = read_HI_data(path)

    A = hi_dict['Bearing3_1_temp'][['pca1']].values
    A -= A.min()-1

    # ts = show_variance_of_increments(A)
    ts = pd.Series(np.diff(np.log(A.T[0])))

    s1_m,s1_sig,s2_m,s2_sig,s3_m,s3_sig, steady_state = parameter_estimation(0,0,ts)

    sp = get_simulated_process(A,A[0, 0],1,A.shape[0],s1_m,s1_sig,s2_m,s2_sig,s3_m,s3_sig,steady_state,1000)

    rul = process_rul(A,1472,1750,sp)

    real_option_analysis(A,sp,5.5,2000,200000,2,10,1450,1750,30,1000)



