#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# ## Adidas Sales Prediction (With Time Series)
# 
# Bu uygulamada Adidas markasına ait satış verileri türünden geleceğe yönelik satış tahmini yapılacaktır

# In[1]:


# Kütüphaneleri içe aktarama
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Sezonluk inceleme ve arima tahmin
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


# ### from statsmodels.tsa.seasonal import seasonal_decompose;
# 
# Bu kütüphane, zaman serisi verilerindeki trend, mevsimsellik ve rassal bileşenleri ayrıştırmak için kullanılır.
# 
# Seasonal_decompose fonksiyonu, aşağıdaki adımları izleyerek zaman serisini bileşenlerine ayırır:
# 
# ** Trend: Zaman serisi verilerindeki uzun vadeli hareket veya genel eğilim.
# ** Mevsimsellik: Belirli bir periyotta tekrarlanan değişimler.
# ** Rassal bileşen: Trend ve mevsimsellikte açıklanamayan, öngörülemeyen dalgalanmalar.
# 
# 
# ### from statsmodels.tsa.arima.model import ARIMA;
# 
# Autoregressive Integrated Moving Average (ARIMA) modelinin Türkçe karşılığı "Otoregresif Bütünleşik Hareketli Ortalama" modeli olarak ifade edilebilir.
# 
# Bu model, zaman serisi verilerini modellemek için kullanılan bir teknik olup, aşağıdaki bileşenlerden oluşur:
# 
# ** Otoregresif (Autoregressive - AR) Bileşen:
# Geçmiş değerlerin mevcut değer üzerindeki etkisini modellemek için kullanılır.
# 
# ** Bütünleşme (Integrated - I) Bileşeni:
# Verinin durağanlık özelliğini modellemek için kullanılır.
# Bazen verinin farkını alma yoluyla durağanlık sağlanır.
# 
# ** Hareketli Ortalama (Moving Average - MA) Bileşeni:
# Geçmiş hata terimlerinin mevcut değer üzerindeki etkisini modellemek için kullanılır.

# In[2]:


ls


# In[3]:


df=pd.read_csv('adidas-quarterly-sales.csv')


# In[4]:


df


# In[5]:


pip install plotly_express


# In[10]:


import plotly_express as px
fig=px.line(df,x='Time Period',y='Revenue', markers=True)
fig.show()


# In[7]:


# İnteraktif veri görselleştirme

result=seasonal_decompose(df['Revenue'],model='multiplicative',period=30)
fig=result.plot()
fig.set_size_inches(15,10)
fig.show()


# Covariance matrix calculated using the outer product of gradients, yani gradyanların dış çarpımı kullanılarak hesaplanan kovaryans matrisi, makine öğrenmesi ve derin öğrenme algoritmalarında sıklıkla kullanılan bir kavramdır.
# 
# Daha detaylı olarak açıklamak gerekirse:
# 
# Gradyan (Gradient): Herhangi bir fonksiyonun türevini (gradyan) hesaplamak, optimizasyon algoritmalarında ve makine öğrenmesi modellerinin eğitiminde kritik öneme sahiptir. Gradyanlar, modelin parametrelerindeki değişimin fonksiyonun değerinde yaptığı değişimi gösterir.
# 
# Dış Çarpım (Outer Product): İki vektörün dış çarpımı, bir matris elde etmek için kullanılan bir işlemdir. Bir vektörün transpozesi ile orijinal vektörün çarpılması sonucu elde edilir.
# 
# Kovaryans Matrisi: Değişkenler arasındaki ilişkiyi gösteren bir matristir. Değişkenlerin birbirleriyle ne kadar ilişkili olduğunu gösterir.

# In[8]:


df


# In[11]:


#Corona olmasaydı
df_without_corona=pd.DataFrame(df['Revenue'][:80],columns=['Revenue'])
df_without_corona                           


# In[17]:


import statsmodels.api as sm 
model=sm.tsa.statespace.SARIMAX(df_without_corona['Revenue'], order=(1, 1, 1),seasonal_order=(1, 1, 1, 4)) # modele satış verisini verdik.
model=model.fit() # modele desenleri öğrettik.
print(model.summary())


# In[18]:


prediction=model.predict(start=len(df_without_corona),end=len(df_without_corona)+100)
print(prediction)


# In[20]:


df_without_corona.plot()
prediction.plot()
plt.show()


# In[21]:


model=sm.tsa.statespace.SARIMAX(df['Revenue'], order=(1, 1, 1),seasonal_order=(1, 1, 1, 4)) # modele satış verisini verdik.
model=model.fit() # modele desenleri öğrettik.
print(model.summary())


# In[22]:


prediction=model.predict(start=len(df),end=len(df)+100)
print(prediction)


# In[23]:


df['Revenue'].plot() 
prediction.plot()
plt.show()


# Covid19 verisini mevsimlikmiş gibi algılayıp her döneme uyguluyor bu aslında modelin bazı noktalarda hata yaptığının göstergesi

# In[ ]:




