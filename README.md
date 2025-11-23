# ⚽ Football AI Grand Chelem - Documentation Complète

## 📝 Description du Projet
Ce projet est une simulation de football 2D avancée, propulsée par une Intelligence Artificielle de pointe. Il ne s'agit pas d'un simple jeu scripté, mais d'un environnement de recherche où des agents apprennent à jouer au football par eux-mêmes, en partant de zéro, jusqu'à développer des stratégies complexes d'équipe.

Le système utilise l'apprentissage par renforcement (Reinforcement Learning) avec l'algorithme **PPO (Proximal Policy Optimization)**, couplé à une architecture **Actor-Critic** et un mécanisme de **Self-Play** (l'IA s'entraîne contre elle-même).

## ✨ Fonctionnalités Clés (Grand Chelem)

### 🧠 Intelligence Artificielle
*   **Algorithme PPO** : Plus stable et performant que le DQN classique.
*   **Architecture Actor-Critic** : Deux réseaux de neurones travaillent en tandem (l'un agit, l'autre juge).
*   **Self-Play** : L'IA joue contre une version "gelée" d'elle-même. Si elle gagne trop souvent, l'adversaire est mis à jour avec la nouvelle version.
*   **Curriculum Learning** : L'IA apprend par étapes (Tir -> Dribble -> Duel 1v1 -> Match 2v2).

### 🤝 Multi-Agents & Communication
*   **Mode 2v2** : Le jeu supporte 4 joueurs (2 par équipe).
*   **Communication** : Les agents possèdent un canal de communication dédié. Ils peuvent envoyer un signal (0-3) à leur coéquipier pour se coordonner (ex: "Je monte", "Passe la balle").

### 🎮 Moteur de Jeu & Physique
*   **Physique Vectorielle** : Mouvements fluides, collisions réalistes.
*   **Effet Magnus** : La balle courbe si elle est frappée avec de l'effet.
*   **Friction de l'Air** : La balle ralentit de manière réaliste.

### 📊 Infrastructure & Contrôle
*   **Entraînement Parallèle** : 16 parties sont jouées simultanément pour accélérer l'apprentissage (optimisé pour Ryzen 5800X).
*   **Web Dashboard** : Une interface de contrôle complète (React + FastAPI) pour suivre les courbes de progression, mettre en pause, ou sauvegarder manuellement.
29: 
30: ### 💾 Gestion Automatique & Optimisation
31: *   **Rotation des Checkpoints** : Le système conserve automatiquement les 15 derniers fichiers de sauvegarde pour éviter de saturer le disque dur.
32: *   **Auto-Tuning Matériel** : Au lancement, le script analyse votre CPU, RAM et GPU pour ajuster automatiquement les paramètres d'entraînement (`BATCH_SIZE`, `PARALLEL_ENVS`, `MAX_MEMORY`) et garantir une stabilité maximale.

---

## 🛠️ Installation

Ce projet a été optimisé pour une machine puissante (Ryzen 5800X + RX 7800 XT).

### Prérequis
*   **Python 3.10+**
*   **Node.js & npm** (pour le dashboard)

### 1. Installation du Backend (Python)
Ouvrez un terminal dans le dossier du projet (`foot/`) :

```bash
# Créer un environnement virtuel (recommandé)
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate  # Sur Linux/Mac
# ou
.\venv\Scripts\activate   # Sur Windows

# Installer les dépendances
pip install pygame torch matplotlib numpy fastapi uvicorn
```

### 2. Installation du Frontend (Dashboard)
Ouvrez un second terminal dans le dossier `foot/dashboard-ui/` :

```bash
cd dashboard-ui
npm install
```

---

## 🚀 Lancement

### Option A : Entraîner l'IA (Mode Principal)
C'est ici que la magie opère. L'IA va jouer des milliers de matchs contre elle-même.

1.  **Lancer l'entraînement et l'API** (Terminal 1, dossier `foot/`) :
    ```bash
    ./venv/bin/python main.py train
    ```
    *Cela va lancer 16 fenêtres de jeu (invisibles ou réduites) et le serveur API sur le port 8000.*

2.  **Lancer le Dashboard** (Terminal 2, dossier `foot/dashboard-ui/`) :
    ```bash
    npm run dev
    ```
    *Ouvrez ensuite votre navigateur sur `http://localhost:5173`.*

**Sur le Dashboard, vous pouvez :**
*   Voir le **Win Rate** (Taux de victoire) et le **Mean Score** (Score moyen).
*   Mettre en pause l'entraînement.
*   Sauvegarder un "Checkpoint" manuellement.
*   Ajuster le "Learning Rate" (vitesse d'apprentissage) en temps réel.

### Option B : Jouer contre l'IA
Une fois que l'IA est forte (après quelques heures), vous pouvez la défier.

```bash
./venv/bin/python main.py play
```
*   **Contrôles** : Flèches directionnelles pour bouger, Espace pour tirer/sprinter.

---

## ⚙️ Configuration Avancée

Le fichier `config.py` contient tous les réglages. Il a été réglé pour votre matériel haut de gamme :

*   `BATCH_SIZE = 8192` : Utilise massivement la VRAM de la RX 7800 XT.
*   `PARALLEL_ENVS = 16` : Utilise tous les cœurs du Ryzen 5800X.
*   `HIDDEN_LAYERS = [2048, 1024, 512, 256]` : Un cerveau très profond pour des stratégies complexes.

Si vous changez de machine pour une moins puissante, réduisez ces valeurs (ex: Batch 1024, Envs 4, Layers [512, 256]).
107: 
108: *Note : Grâce à l'Auto-Tuning, ces valeurs sont désormais ajustées automatiquement au démarrage si nécessaire.*

## 🐛 Dépannage

*   **Erreur "Address already in use" (Port 8000)** : Si vous relancez l'entraînement trop vite, le port de l'API peut être encore occupé. Attendez quelques secondes ou tuez le processus python (`pkill python`).
*   **L'IA ne bouge pas** : Au tout début, c'est normal, elle explore. Attendez que l'Epsilon (taux d'aléatoire) diminue.
*   **Dashboard vide** : Vérifiez que le script python (`main.py train`) tourne bien et n'a pas planté.

---

**Bon entraînement ! 🏆**
