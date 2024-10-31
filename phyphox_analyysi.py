import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from geopy.distance import geodesic
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import folium
import os
from scipy.fft import fft, fftfreq
import streamlit as st 

# Lataa dataa
acc_data = pd.read_csv('Linear Accelerometer.csv')
loc_data = pd.read_csv('Location.csv')

# Suodatusfunktio
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Suodata kiihtyvyysdata
filtered_acc = butter_lowpass_filter(acc_data['Z (m/s^2)'], cutoff=3, fs=50)
acc_data['filtered_z'] = filtered_acc

# Laske askelmäärä suodatetusta datasta
peaks, _ = find_peaks(filtered_acc, height=0.1, distance=20)
step_count_filtered = len(peaks)
print("Lasketut askeleet suodatetusta datasta:", step_count_filtered)

# Tee Fourier-muunnos ja laske taajuudet ja tehospektrin tiheys
acc_fft = fft(acc_data['Z (m/s^2)'])
freqs = fftfreq(len(acc_fft), d=1/50)  # näytteenottotaajuus 50 Hz
psd = np.abs(acc_fft) ** 2

# Laske askelmäärä Fourier-analyysin perusteella
# Rajaa Fourier-analyysi vain kävelytaajuusalueelle, esimerkiksi 1–3 Hz
relevant_freqs = (freqs >= 1) & (freqs <= 3)  # Rajaa taajuudet 1–3 Hz alueelle
step_peaks, _ = find_peaks(psd[relevant_freqs], height=0.1, distance=1)
step_count_fourier = len(step_peaks)
print("Lasketut askeleet Fourier-analyysin perusteella:", step_count_fourier)

# Laske kuljettu matka ja keskiarvoinen nopeus
total_distance = sum(
    geodesic((loc_data['Latitude (°)'][i], loc_data['Longitude (°)'][i]),
             (loc_data['Latitude (°)'][i+1], loc_data['Longitude (°)'][i+1])).meters
    for i in range(len(loc_data)-1)
)
average_speed = total_distance / (loc_data['Time (s)'].iloc[-1] - loc_data['Time (s)'].iloc[0])

step_length = total_distance / step_count_filtered
print("Askelpituus:", step_length, "metriä")

# Varmista, että output-kansio on olemassa
if not os.path.exists("output"):
    os.makedirs("output")

# Piirrä kuvaaja suodatetusta kiihtyvyydestä ja tallenna se
plt.figure()
plt.plot(acc_data['Time (s)'], acc_data['filtered_z'])
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys (m/s^2)")
plt.title("Suodatettu kiihtyvyysdata")
plt.savefig("output/filtered_acceleration.png") 
plt.close()

# Piirrä tehospektrin tiheys ja tallenna se
plt.figure()
plt.plot(freqs[:len(freqs)//2], psd[:len(psd)//2])  
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Tehospektrin tiheys")
plt.title("Tehospektrin tiheys Fourier-analyysin perusteella")
plt.savefig("output/power_spectral_density.png") 
plt.close()
print("Tehospektrin kuva tallennettu output/power_spectral_density.png")

# Luo kartta
m = folium.Map(location=[loc_data['Latitude (°)'].mean(), loc_data['Longitude (°)'].mean()], zoom_start=15)
route = list(zip(loc_data['Latitude (°)'], loc_data['Longitude (°)']))
folium.PolyLine(route, color="blue", weight=2.5, opacity=1).add_to(m)
m.save("output/route_map.html")

# Streamlit-käyttöliittymä
st.title("Liikunnan analyysi")
st.write(f"Lasketut askeleet suodatetusta datasta: {step_count_filtered}")
st.write(f"Lasketut askeleet Fourier-analyysin perusteella: {step_count_fourier}")
st.write(f"Kuljettu matka: {total_distance:.2f} metriä")
st.write(f"Keskinopeus: {average_speed:.2f} m/s")
st.write(f"Askelpituus: {step_length:.2f} metriä")

# Näytä suodatettu kiihtyvyysdata
st.image("output/filtered_acceleration.png", caption='Suodatettu kiihtyvyysdata')

# Näytä tehospektri, jos kuva on tallennettu
if os.path.exists("output/power_spectral_density.png"):
    st.image("output/power_spectral_density.png", caption='Tehospektrin tiheys')
else:
    st.error("Tehospektrin kuvaa ei löytynyt!")

# Näytä kartta
html_file = os.path.join("output", "route_map.html")
if os.path.exists(html_file):
    st.subheader("Reittikartta")
    with open(html_file, "r") as f:
        html_content = f.read()
        st.components.v1.html(html_content, height=500, scrolling=True)
else:
    st.error("HTML-tiedostoa ei löytynyt!")
