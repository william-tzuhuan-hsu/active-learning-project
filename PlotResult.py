import matplotlib.pyplot as plt
def myplot(result_rand, result_div, result_uns, n_init, batchsize, eval):
    if eval == 'MSE':
        rand_mean, rand_std=result_rand.MSE()
        div_mean, div_std=result_div.MSE()
        uns_mean, uns_std=result_uns.MSE()

        plt.ylabel('Mean squared error on the unobserved set')

    elif eval == 'CV':
        rand_mean, rand_std=result_rand.CV()
        div_mean, div_std=result_div.CV()
        uns_mean, uns_std=result_uns.CV()

        plt.ylabel('Cross Validation Error on the training set')
    else:
        raise Exception('evaluation method error')
    n=len(rand_mean)
    x_axis=list(range(n_init,n_init+n*batchsize,batchsize))

    plt.plot(x_axis,uns_mean,color='tab:orange',label='Uncertainty Sampling')
    plt.legend(loc ="upper right")
    plt.errorbar(x_axis,uns_mean,yerr = uns_std,fmt ='o',color='tab:orange',markersize=1,alpha=0.5)

    plt.plot(x_axis,rand_mean,color='tab:blue',label='Random')
    plt.legend(loc ="upper right")
    plt.errorbar(x_axis,rand_mean,yerr = rand_std,fmt ='o',color='tab:blue',markersize=1,alpha=0.5)

    plt.plot(x_axis,div_mean,color='tab:red',label='Diversity Sampling')
    plt.errorbar(x_axis,div_mean,yerr = div_std,fmt ='o',color='tab:red',markersize=1,alpha=0.5)
    plt.legend(loc ="upper right")

    plt.xlabel('number of instances observed')
    plt.show()
        

