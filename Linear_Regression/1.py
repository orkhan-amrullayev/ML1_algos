{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "LinReg_1.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "R-bgLBn-d74N"
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CeRkPH3wePi9"
      },
      "source": [
        "X = 2 * np.random.rand(100, 1)\n",
        "y = 4+3 * X + np.random.rand(100, 1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 281
        },
        "id": "E4QPp5VYeZGE",
        "outputId": "c7368e65-c2fb-43de-cb56-28be16b26c61"
      },
      "source": [
        "# plt.plot(X, y, \"b\")   ## it is line plot\n",
        "plt.plot(X, y, \".b\")  #scatter plot"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7f4b0707f950>]"
            ]
          },
          "metadata": {},
          "execution_count": 6
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAYpElEQVR4nO3df5BddXnH8c+TDYRCrcBmtVRYA1NHq1IKveO4FO3a0Ar4AzQtA1NnQ0Q32mjZOtaSYWgZMy2205mmHTqtGwSTqUUtiLW22ii6hakLssFoUEQBIYaiWeNvLAvJPv3jnNuc3Nwf557zPefec+/7NZO5d889956Hk8uz3zzfX+buAgBUz4peBwAAyIYEDgAVRQIHgIoigQNARZHAAaCiVpZ5sdWrV/uaNWvKvCQAVN6uXbu+5+5jjcdLTeBr1qzRwsJCmZcEgMozs8eaHaeEAgAV1TGBm9lNZrbfzO5PHPs9M/uqmS2bWa3YEAEAzaRpgX9Q0gUNx+6X9EZJd4YOCACQTscauLvfaWZrGo49IElmVkxUAICOCq+Bm9m0mS2Y2cLi4mLRlwOAoVF4Anf3WXevuXttbOyoUTAAgIwYhQIABZmfl66/PnosQqnjwAFgWMzPS2vXSk8/LR17rHTHHdLERNhrpBlGeIukeUkvNLN9Znalmb3BzPZJmpD072b2n2HDAoBqm5uLkvehQ9Hj3Fz4a6QZhXJ5i5duDxwLAAyMycmo5V1vgU9Ohr8GJRQAyGB+PmpVT042L41MTERlk3bn5EUCB4Aupa1vT0wUk7jrGIUCAF1KU98uegSKRAscALrWqb5dxggUiQQOAF2p1763bpUOHGhe327WQieBA0APpW1ZlzECRSKBA0BqaVvWZYxAkUjgAJBaNy3rokegSCRwAEitrJZ1WiRwAIh1mpwjldOyTosEDgAqb+hfSEzkAQCVs/hUaCRwANDhDsqRkWKH/oVECQUAlL+DMk39PDQSOADE0nRQNkvUvaqfk8ABIKVWibrTBJ+iWuckcABIqVWibjfBp8jWOQkcAFJqlajb1c+LXNiKBA4AKbVL1K3q50UubEUCB4AudDsTs8jp9x0TuJndJOm1kva7+0vjYydL+oikNZIelXSpu/8gXFgA0DudOh277ZQsavp9mhb4ByXdIGlH4tjVku5w9/eZ2dXxz38SPjwAKFenTsd+mnLfcSamu98p6fsNhy+WtD1+vl3SJYHjAoCe6DSlvpsp90Xvi5m1Bv5cd38ifv4dSc9tdaKZTUualqTx8fGMlwOAcnTqdEzbKVlGSz13J6a7u5l5m9dnJc1KUq1Wa3keAPSL9eujx6mpo5Nu2k7JMvbFzJrAv2tmp7j7E2Z2iqT9IYMCgF5obDVPTTU/L02nZBn7YmZdjfATkuLfUVov6V/DhAMAvRNySdl6S33LluI6OtMMI7xF0qSk1Wa2T9KfSXqfpI+a2ZWSHpN0afjQAAyrXqzsJ4VvNRe9e0/HBO7ul7d4aW3gWACgaeefVE5C77c9LzthJiaAvpIsYywtSTMz0u7d0c9ljLvupz0vO2FHHgB9pV7GWLFCWl6W7r23eludlYUEDqCv1MsY558fJXGPBx+bVWers7KQwAH0nYkJ6brrpFWroj0qV62SNm6sxk7xZaIGDqAvVa1DsRdI4AD6VpU6FHuBEgqAoVb0glNFogUOYOCknQjUT0vDZkECBzBQuknKZSw4VSRKKAAGSjfrmdTHnI+MVHOIIi1wAAOlm/VMqj7ShQQOYKB0m5SrPNKFBA6gcGWvLljlpNwNEjiAQhUx0qNXy832GxI4gEKFHulR9aF/ITEKBUChQo/0CLlrTtXRAgdwlJAlitAjPcrYa7IqSOAAjlBEiSJkp2LVh/6FRAIHhlSrVnYVZicOyyiTTkjgwBBq18qmRFEddGICQ6hdR2C9RLFlS3flkyqv6ldVuVrgZnaVpLdKMknb3H1rkKgAFKpTK7vbEgVD+3ojcwvczF6qKHm/TNJZkl5rZr8cKjAAxcnaym4l69A+Wu355GmB/4qke9z9Z5JkZv8l6Y2S/ipEYACKFbIjMEvdnFZ7fnlq4PdLeoWZjZrZ8ZIuknRa40lmNm1mC2a2sLi4mONyAPpVlhY9E3Lyy9wCd/cHzOwvJe2U9KSk3ZIONTlvVtKsJNVqNc96PQD9rdsWPaNd8svVienuH5D0AUkys7+QtC9EUAAGHxNy8ss7CuU57r7fzMYV1b9fHiYsAIOg05R8JuTkk3ciz21mNirpGUmb3P2HAWICMACSnZQjI9Kb3yxNTZGwQ8o1kcfdX+HuL3b3s9z9jlBBAai+xk7K978/SugMGQyHmZgAClHvpDSLfnZntEloJHAAhah3Um7cKK1aVd2d3/sZi1kBKEy9k3JqitEmRSCBAwOqn/aNZLRJMUjgwABimvpwoAYOVFCnRaCYpj4caIEDFZAsh0idW9dMUx8OJHCgR9LWqBvLIevXd97yjGnqw4EEDvRANzXqxnKIFA3JW16OHlu1ruk4HHzUwIEeaEzKO3a0rmnXyyH1cdRnn314ckz9EcOJFjhQsGalkmSNemREuvlm6eDB5q3xxnLI3Fx0rnv02I+7xqMcJHCgQK1KJcmkvHevtG1b55p28hgdlJBI4EChmg3nqyfielKen5e2b0+fkOmgRB0JHChQmuF8WRIyHZSQSOBAodImZxIysiCBAzmkGcvdi+TcT+ugoDgkcCCjfltvpJ60R0elmZn+iQvFIYEDGbXroCxb8pfJihVRTMvLvY8LxWIiD5BR4wSbVqNHOi08FcLcnLS0FCXuQ4eiJM4GCoOPFjiQUZoOyrLKLKOjUYtbih7f/W7pxBOpgQ+6XAnczP5I0lskuaQ9kja4+1MhAgOqoFMHZVlllgMHolb38nL0eOKJ0ubN4a+D/pK5hGJmz5P0h5Jq7v5SSSOSLgsVGDAI0pZZpHyllsnJw/tOrlpF2WRY5C2hrJT0c2b2jKTjJf1P/pCAwZF2HHjeUguzM4dT5gTu7o+b2V9L2ivpfyXtdPedjeeZ2bSkaUkaHx/PejmgstKMA09bamk3vpvJQMMnTwnlJEkXSzpd0i9JOsHM3tR4nrvPunvN3WtjY2PZIwUGWJpSS72Vfu210WORo1pQDXmGEZ4v6Vvuvujuz0j6mKRzw4QFDJd6CWTLltblE/a5RKM8NfC9kl5uZscrKqGslbQQJCpgCHUqgbDPJRrlqYHfY2a3SrpP0kFJX5I0GyowAEeioxKNzN1Lu1itVvOFBRrpANANM9vl7rXG40ylB0pWxtR6DAem0gMl6rcVDFFttMAxEKrSqmUkCUKiBY7Kq1KrlpEkCIkEjsrrp3W5O2EkCUIigaPyqtaqZco7QiGBo/Jo1WJYkcAxEMpq1Sb3nTxwgF8Y6C0SOIZO1h3bZ2elTZuiWrt7tHHCqlXtO03ZHR5FIoFjoHRKmFlHrMzPS+94h3Tw4OFjnTYNrtLoGFQT48AxMNIst5p1HPbcXPSepBUr2neaMuYbRaMFjoGRZjhh1hEr9S3LlpaiNbsvvVRaXJTWrWvdqq7a6BhUDwkcldOqTJImYWYdsZJ83+ioNDMTXeeuu6Qzz2z+OYyOQdFI4KiUdnXltAkz64iV+vve/nbpqaeijsylpfYThxjzjSKRwFEpncokRSfM+Xnpxhuj5C1FHZmjo8VdD2iHTkyULs/CU2n2jixSs87MAwfKjQGoowWOUuUdWtfruvLkpHTMMVH8UtSxSeckeoUEjlKFWHiql3XliYko5h07op+npqhxo3dI4CjN/Ly0d29U/pBal0D6ffYiHZPoFyRwlCJZOlm5UnrrW5u3Xpm9CKSXuRPTzF5oZrsTf35sZjMhg8PgSJZODh6UxsebJ2ZmLwLpZW6Bu/uDkn5NksxsRNLjkm4PFBcGTNpZiUXMXuz3kgyQVagSylpJD7v7Y4E+DwOm1eiRxuQaepQJJRkMslAJ/DJJtwT6LAyoxs6/Vsm1VSdhlpZ0lbZbA7qVO4Gb2bGSXi9pc4vXpyVNS9L4+Hjey2GApE2u8/PRsL2bb47q5920pFlQCoMsRAv8Qkn3uft3m73o7rOSZiWpVqt5gOthQKRJrvVWen3tEam7lnSvJ/4ARQqRwC8X5RNkkCa51lvpnvjV321LmnHbGFS5EriZnSDptyVtDBMOhk2n5Do5GW2cUF9/ZGRE2rqVhAxIORO4uz8pibXYBlyvh+EtLx9+7s7iUUAdMzHRVq+H4c3NHVk+GRmhIxKoYzlZtNXrmZH1rcxWrIim4N9wA+UToI4WONoKPQyv23IMo0iA1kjgaKtxL8h6CzxLIs1ajmEUCdAcCRwd1ZNn3lp4shzz1FPR5BwSM5AdNXAcpdmWZyFq4ZOTUR1bijomb7op27ZqACIk8CGSZi/Kepnj2mujx/q57faiTLvH5cSEtGGDZBb9fOgQy8UCeVBCGRJp68+tyhztVhPsprQyNRW1vJ95hiGBQF60wIdE2hJIuzLHxIS0efORCbrd57Zqmddb4PVHANmQwIdEuxJIUrdljlaf26oUMzcXrSjoHj1SQgGyI4EPiXoJZMuWdGWO447rnOzbfW6rlnnaXyQAOjP38lZ4rdVqvrCwUNr1kF3e9U/a1cZ7vbYKUDVmtsvda0cdJ4GjKCRqIIxWCZxRKCgMMyiBYlEDB4CKIoGjUGkn+QDoHiUUFKbXa4kDg44WOArT67XEgUFHAh9iRZc3GPMNFIsSypAqo7zBZgxAsUjgQ6pZeaOIBMtQQqA4uUooZnaimd1qZl83swfMjP9VC1BEqYPyBlB9eVvgfyvp0+7+u2Z2rKTjA8SEhKJKHZQ3gOrLnMDN7NmSXinpCkly96clPR0mrOHVOP28yFIH5Q2g2vK0wE+XtCjpZjM7S9IuSVe5+5PJk8xsWtK0JI2Pj+e43OBr1toOvSs8gMGRpwa+UtI5kv7B3c+W9KSkqxtPcvdZd6+5e21sbCzH5QZfq9b21q1RYt+6NXuLmRmRwODJ0wLfJ2mfu98T/3yrmiRwpNestT0/L83MRMfuuks688zukzgzIoHBlLkF7u7fkfRtM3thfGitpK8FiWrAdLPpb+PmCCFmMzIjEhhMeUehvFPSh+IRKI9I2pA/pMHSbeu3sWMxRA2cOjowmHIlcHffLemoRcaHRbMNC2Znpdtuk9atk6an848iCTHcjyGDwGBiJmZGzVrWe/ZIGzdGr+/cGT22qmt3k0w7Dfdr93nJ1zZv7nw+gOoggWfUrGX98Y8fec5tt0Wt8GTrVwrbodhp78nG10JfH0DvsBphRo1T0UdHpfvuO/Kcdeuix4mJqPUbqlMyqd3nNXuNDk1gcNACz6ixrjw3JyX3h77kkqj13Sh0h2K7z2v1Gh2awGAggefQWJtOJsb3vKf1e0J2KLb7vFav0aEJDAbzZLOxYLVazRcWFkq7Xtnm56UdO6LnU1MkRwBhmNkudz9qxB818MC2b5e2bYs6Cpm2DqBIJPAGedYMoYMQQJmogSfkXTOEGY8AykQCT+iHWZMAkBYJPCFEC5pNEgCUhQSeQAsaQJWQwBuU1YJmPRIAeZHAe4ANFgCEwDDCHmC4IYAQSOA90LgQFsMNAWQxsCWUvDXmImvUdJYCCGEgE3jeGnMZNWqGGwLIayBLKHlrzNSoAVTBQCbwvDVmatQAqiBXCcXMHpX0E0mHJB1sttxhL+StMVOjBlAFudYDjxN4zd2/l+b8Xq4HzsQZAFXVaj3wgezEbNTYKbl1q3TgAMkcQLXlTeAuaaeZuaT3u/ts4wlmNi1pWpLGx8dzXi6bZKfk0pK0aVO0fyWzIAFUWd5OzPPc/RxJF0raZGavbDzB3WfdvebutbGxsZyXyybZKTkyIi0vM8IEQPXlaoG7++Px434zu13SyyTdGSKwkJKdkqOj0sxM6yVjqZUDqIrMCdzMTpC0wt1/Ej//HUnvDRZZQqekmibpJifOnHlm8/NZZApAleRpgT9X0u1mVv+cf3b3TweJKqFZUpUOJ2Cp+6TbahZk3h15AKBMmRO4uz8i6ayAsTTV2AE5MyPt3h39fOyx0vr14ZIue1oCqJK+H0ZYT6pLS1Hn4733RiNIpCjRSuGSLhN4AFRJrok83co6kWd+XrruOumzn42SuCSZSccdd3RJhaQLYNBUeiLPxESUwO+6K2ppr1wpbdggTU0dTtgkbgDDphIJXKK8AQCNKpPAJdbQBoCkgVxOFgCGQaUT+Py8dP310SMADJtKlVCSmDUJYNhVtgXOtmcAhl1lEzjbngEYdpUtoTCsEMCwq2wClxhWCGC4VbaEAgDDjgQOABVVuQTO2G8AiFSqBs7YbwA4rFItcMZ+A8BhlUrgjP0GgMMqVUKpj/3esaPXkQBA71WqBV63fbu0bVtUD6czE8CwqlwCpw4OAJHcCdzMRszsS2b2yRABdUIdHAAiIWrgV0l6QNIvBPisjlgDBQAiuRK4mZ0q6TWS/lzSu4JElAJroABA/hLKVknvkbTc6gQzmzazBTNbWFxczHk5AEBd5gRuZq+VtN/dd7U7z91n3b3m7rWxsbGslwMANMjTAv8NSa83s0clfVjSb5nZPwWJCgDQUeYE7u6b3f1Ud18j6TJJn3P3NwWLDADQVuXGgQMAIkGm0rv7nKS5EJ8FAEjH3L28i5ktSnosw1tXS/pe4HBC6dfYiKs7/RqX1L+xEVd38sT1fHc/ahRIqQk8KzNbcPdar+Nopl9jI67u9GtcUv/GRlzdKSIuauAAUFEkcACoqKok8NleB9BGv8ZGXN3p17ik/o2NuLoTPK5K1MABAEerSgscANCABA4AFdXzBG5mF5jZg2b2kJld3eT1VWb2kfj1e8xsTeK1zfHxB83s1SXH9S4z+5qZfcXM7jCz5ydeO2Rmu+M/nyg5rivMbDFx/bckXltvZt+M/6wPGVfK2P4mEdc3zOyHidcKuWdmdpOZ7Tez+1u8bmb2d3HMXzGzcxKvFX2/OsX2+3FMe8zsC2Z2VuK1R+Pju81soeS4Js3sR4m/rz9NvNb2O1BwXH+ciOn++Dt1cvxakffrNDP7fJwPvmpmVzU5p5jvmbv37I+kEUkPSzpD0rGSvizpxQ3n/IGkf4yfXybpI/HzF8fnr5J0evw5IyXG9SpJx8fP316PK/75pz28X1dIuqHJe0+W9Ej8eFL8/KQyY2s4/52Sbirhnr1S0jmS7m/x+kWSPiXJJL1c0j1l3K+UsZ1bv6akC+uxxT8/Kml1j+7ZpKRP5v0OhI6r4dzXKVqfqYz7dYqkc+Lnz5L0jSb/XxbyPet1C/xlkh5y90fc/WlFqxpe3HDOxZK2x89vlbTWzCw+/mF3X3L3b0l6KP68UuJy98+7+8/iH++WdGqga+eKq41XS/qMu3/f3X8g6TOSLuhhbJdLuiXg9Zty9zslfb/NKRdL2uGRuyWdaGanqPj71TE2d/9CfG2pvO9YmnvWSp7vZ+i4Svl+SZK7P+Hu98XPf6Joh7LnNZxWyPes1wn8eZK+nfh5n47+D///c9z9oKQfSRpN+d4i40q6UtFv17rjLNrE4m4zuyRQTN3EtS7+Z9qtZnZal+8tOjbF5abTJX0ucbioe9ZJq7iLvl/davyOuaSdZrbLzKZ7EM+EmX3ZzD5lZi+Jj/XFPTOz4xUlwdsSh0u5XxaVeM+WdE/DS4V8z4IsZjXMzOxNkmqSfjNx+Pnu/riZnSHpc2a2x90fLimkf5N0i7svmdlGRf96+a2Srp3WZZJudfdDiWO9vGd9zcxepSiBn5c4fF58v54j6TNm9vW4hVqG+xT9ff3UzC6S9HFJLyjp2mm8TtJ/u3uytV74/TKzn1f0S2PG3X8c8rNb6XUL/HFJpyV+PjU+1vQcM1sp6dmSDqR8b5FxyczOl3SNpNe7+1L9uLs/Hj8+omiVxrPLisvdDyRiuVHSr6d9b9GxJVymhn/eFnjPOmkVd9H3KxUz+1VFf48Xu/uB+vHE/dov6XaFKx925O4/dvefxs//Q9IxZrZafXLP1P77Vcj9MrNjFCXvD7n7x5qcUsz3rIiifhfF/5WKivan63Cnx0saztmkIzsxPxo/f4mO7MR8ROE6MdPEdbaiDpsXNBw/SdKq+PlqSd9UoI6clHGdknj+Bkl3++HOkm/F8Z0UPz+5zL/L+LwXKepQsjLuWfyZa9S6Q+41OrJz6Ytl3K+UsY0r6ts5t+H4CZKelXj+BUkXlBjXL9b//hQlwr3x/Uv1HSgqrvj1Zyuqk59Q1v2K/9t3SNra5pxCvmdBv4wZ/+MvUtRr+7Cka+Jj71XUqpWk4yT9S/xF/qKkMxLvvSZ+34OSLiw5rs9K+q6k3fGfT8THz5W0J/7y7pF0ZclxXS/pq/H1Py/pRYn3vjm+jw9J2lD232X883WS3tfwvsLumaKW2BOSnlFUX7xS0tskvS1+3ST9fRzzHkm1Eu9Xp9hulPSDxHdsIT5+Rnyvvhz/XV9TclzvSHzH7lbiF0yz70BZccXnXKFocEPyfUXfr/MU1di/kvi7uqiM7xlT6QGgonpdAwcAZEQCB4CKIoEDQEWRwAGgokjgAFBRJHAAqCgSOABU1P8Bo031qlPk2ZkAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U-ZT6DaZedoc"
      },
      "source": [
        "from sklearn.linear_model import LinearRegression"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q_ef7zwCe0Lz"
      },
      "source": [
        "model = LinearRegression()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ea3DUqcwfmYO",
        "outputId": "77f54c9d-cb4b-48f3-c3c6-73d483e66a17"
      },
      "source": [
        "model.fit(X, y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rKoZQxqpgges"
      },
      "source": [
        "## Shows X - as we have just one indeendent variable, amount of output is 1"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e5Hsj30nf_Pm",
        "outputId": "08a5a07d-94e5-442d-ae18-afd1cef23ad8"
      },
      "source": [
        "model.coef_"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[2.90277363]])"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PMpaW3whgt-o"
      },
      "source": [
        "## For intercept"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fKjNeKr2gtsc",
        "outputId": "c376354c-e097-4516-ea4a-8f702857adcb"
      },
      "source": [
        "model.intercept_"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([4.6109199])"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TbKPLLFPg1TA"
      },
      "source": [
        "## both together"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-I9EN0vmgd_o",
        "outputId": "bda3c7a3-5f7a-4453-c0ff-4debf15cd7ff"
      },
      "source": [
        "model.coef_, model.intercept_"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(array([[2.90277363]]), array([4.6109199]))"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "R-XCum_mg6Rz"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
