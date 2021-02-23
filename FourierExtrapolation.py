import numpy as np
import pylab as pl
from numpy import fft
from util import mean_absolute_percentage_error, mean_squared_error, get_data


def fourierExtrapolation(x, n_predict, n_harm):
    n = x.size
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1) # find linear trend in x that is why degree is 1
    x_notrend = x - p[0] * t  # de-trended x
    x_freqdom = fft.fft(x_notrend)  # de-trended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t # re-trending


def main():

    df = get_data()
    df = df.set_index('Sample')
    y = df['error']
    train = y.iloc[:-30]
    test = y.iloc[-30:]

    extrapolation = fourierExtrapolation(train, 30, 4)
    extrapolation_20tones = fourierExtrapolation(train, 30, 10)

    test_predictions = extrapolation[-30:]
    test_predictions_20 = extrapolation_20tones[-30:]
    mse = mean_squared_error(test, test_predictions)
    mape = mean_absolute_percentage_error(test, test_predictions)
    mse20 = mean_squared_error(test, test_predictions_20)
    mape20 = mean_absolute_percentage_error(test, test_predictions_20)
    print('Mape: ', mape)
    print('Accuracy is :', 100 - mape, '%')
    print('The Mean Squared Error of our forecasts is {}'.format(mse))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
    print('----------------------------------------------------------------------------------')
    print('Mape 10 harmonics: ', mape20)
    print('Accuracy with 10 harmonics is :', 100 - mape20, '%')
    print('The Mean Squared Error of 10 harmonics forecasts is {}'.format(mse20))
    print('The Root Mean Squared Error of 10 harmonics forecasts is {}'.format(round(np.sqrt(mse20), 2)))

    pl.title('Graph starting from 350k tightening')
    pl.xlabel('Sampled Tightening (1 sample = 500 Tightening)')
    pl.ylabel('Number of NOKs')
    pl.plot(np.arange(0, y.size), y, 'g', label='actual', linewidth=3)
    pl.plot(np.arange(0, train.size), train, 'b', label='train', linewidth=3)
    pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation 4 harmonics')
    pl.plot(np.arange(0, extrapolation_20tones.size), extrapolation_20tones, 'k', label='extrapolation 10 harmonics')
    pl.legend()
    pl.show()


if __name__ == "__main__":
    main()
