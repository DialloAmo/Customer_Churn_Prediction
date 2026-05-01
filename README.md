# 📊 Prédiction du Churn Client & Recommandations Business

## 🎯 Objectif du projet

Ce projet a pour objectif de **prédire le churn client** (résiliation) et surtout de **transformer les résultats du modèle en décisions concrètes pour l’entreprise**.

L’analyse répond à trois questions clés :

* **Qui est à risque de churn ?**
* **Qu'est ce qui l'explique ?**
* **Quelles actions actions entreprendre pour reduire le churn ?**

---

## 📂 Structure du projet

```
churn_prediction/
│
├── data/
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_eda_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_interpretation_business.ipynb
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── modeling.py
│   ├── evaluation.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│
├── models/
└── README.md
```

---

## 📊 Données

Le dataset contient des informations clients d’une entreprise de télécommunications :

* données démographiques
* type de contrat
* services souscrits
* facturation et paiement

Variable cible :

* **Churn (Yes / No)**

---

## 🔍 Analyse exploratoire (EDA)

Principaux enseignements :

* Les clients récents sont plus susceptibles de résilier
* Les contrats mensuels sont fortement associés au churn
* Les clients avec peu de services présentent un risque plus élevé
* Certains modes de paiement sont liés à un churn plus important

---

## ⚙️ Modélisation

### Modèles testés :

* Dummy Classifier (baseline)
* Régression Logistique ✅ (modèle retenu)
* Random Forest
* XGBoost

### Modèle final :

👉 **Régression Logistique avec class_weight="balanced"**

### Pourquoi ce choix ?

* Bon niveau de performance
* Capacité à détecter les churners (recall élevé)
* Modèle interprétable
* Résultats stables

---

## 📈 Évaluation du modèle

Principales métriques :

* **Recall ≈ 0.90** → capacité à détecter les clients qui churnent
* **F1-score ≈ 0.60** → compromis entre précision et rappel
* ROC-AUC / PR-AUC
* Brier score (calibration)

---

## 🎯 Optimisation du seuil de décision

Le seuil par défaut (0.5) n’a pas été conservé.

👉 j'ai retenu un seuil personnalisé de $0.36$ en fonction du compromis précision / rappel.

### Logique métier :

* Il est plus coûteux de **ne pas détecter un client qui va churner**
* j'ai privilégié donc le **sensibilité (recall)**

---

## 🧠 Interprétation du modèle

### 🔹 Interprétation globale

Les principaux facteurs de churn sont :

* une faible ancienneté
* un contrat mensuel
* un faible nombre de services

👉 Ces éléments traduisent un **faible engagement client**

---

### 🔹 Interprétation locale

Exemple de clients à risque :

* clients récents
* contrats mensuels
* peu de services

👉 Profil typique : client peu engagé, susceptible de résilier facilement

---

### 🔹 Importance des variables

L’analyse confirme que :

* l’ancienneté
* le type de contrat
* les services souscrits

sont les principaux leviers explicatifs du churn.

---

## ⚙️ Variables actionnables

| Type                       | Variables                          |
| -------------------------- | ---------------------------------- |
| Non actionnables           | genre, ancienneté passée           |
| Partiellement actionnables | mode de paiement, facturation      |
| Actionnables               | type de contrat, services, support |

---

## 🚀 Recommandations business

### 🔹 Qui est à risque ?

* clients récents
* clients avec contrat mensuel
* clients faiblement engagés

---

### 🔹 Pourquoi ?

* faible engagement
* absence de contrainte contractuelle
* faible dépendance aux services

---

### 🔹 Que faire ?

* 🎯 cibler en priorité les clients à haut risque
* 📉 encourager le passage à des contrats longue durée
* ➕ proposer des services additionnels accessibles
* 💬 améliorer le support client et l'assistance technique
* 📊 utiliser le score de churn pour prioriser les actions commerciales

👉 Les actions doivent prioritairement cibler les clients à **fort risque et forte valeur**.

---

## 🧠 Conclusion

Ce projet montre comment un modèle de machine learning peut être utilisé comme un **outil d’aide à la décision**, permettant de :

* identifier les clients à risque
* comprendre les causes du churn
* mettre en place des actions de rétention ciblées

---

## 🛠️ Technologies utilisées

* Python (Pandas, NumPy)
* Scikit-learn
* Matplotlib / Seaborn
* SHAP
* Jupyter Lab

---

## 👤 Auteur

Amadou Diallo
Futur Data Scientist / ML Engineer
https://github.com/DialloAmo

---

## 📌 Perspectives

* Déploiement du modèle (API / dashboard)
* Intégration du coût métier
* Amélioration des features
