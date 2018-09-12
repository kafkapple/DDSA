import matplotlib
matplotlib.use('Agg')

import elice_utils
import matplotlib.pyplot as plt
import random
import numpy as np

def getStatistics(data) :
    '''
    추출 결과가 리스트로 주어질 때, 추출된 결과의 평균과 분산을 반환하는 프로그램을 작성하세요.
    '''
    average =  np.mean(data)
    variance = np.var(data)
    

    return (average, variance)

def doGaussianSample(mu, sigma, n) :
    '''
    평균이 mu, 분산이 sigma^2 인 가우시안 분포로부터 n개의 표본을 추출하여 반환하는 함수를 작성합니다.
    '''    

    result = np.random.normal(mu,sigma,n)

    return result

def plotResult(data) :
    '''
    숫자들로 이루어진 리스트 data가 주어질 때, 이 data의 분포를 그래프로 나타냅니다.

    이 부분은 수정하지 않으셔도 됩니다.
    '''

    frequency = [ 0 for i in range(int(max(data))+1) ]
    
    for element in data :
        frequency[int(element)] += 1

    n = len(frequency)

    myRange = range(1, n)
    width = 1

    plt.bar(myRange, frequency[1:])

    plt.xlabel("Sample", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)

    filename = "chart.svg"
    plt.savefig(filename)
    elice_utils.send_image(filename)

    plt.close()

def main():
    '''
    이 부분은 수정하지 않으셔도 됩니다.
    '''

    line = [int(x) for x in input("입력 > ").split()]
    
    mu = line[0]
    sigma = line[1]
    n = line[2]

    result = doGaussianSample(mu, sigma, n)

    plotResult(result)
    
    stat = getStatistics(result)

    print("average : %.2lf" % stat[0])
    print("variance : %.2lf" % stat[1])

if __name__ == "__main__":
    main()
