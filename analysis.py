import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def detect_fixations(gaze_data, dispersion_threshold=50, duration_threshold_ms=100):
    """Detects fixations using a simple dispersion-based algorithm and calculates related metrics."""
    fixations = []
    current_fixation_start_index = 0

    if not gaze_data:
        return {'num_fixations': 0, 'avg_fixation_duration': 0, 'gaze_dispersion_x': 0, 'gaze_dispersion_y': 0}

    for i in range(len(gaze_data)):
        window_points = gaze_data[current_fixation_start_index : i + 1]
        if not window_points: continue

        min_x = min(p.get('x', 0) for p in window_points)
        max_x = max(p.get('x', 0) for p in window_points)
        min_y = min(p.get('y', 0) for p in window_points)
        max_y = max(p.get('y', 0) for p in window_points)

        x_dispersion = max_x - min_x
        y_dispersion = max_y - min_y
        total_dispersion = max(x_dispersion, y_dispersion)

        start_time = window_points[0].get('timestamp', 0)
        end_time = window_points[-1].get('timestamp', 0)
        duration = end_time - start_time

        if total_dispersion > dispersion_threshold:
            potential_fixation_window = gaze_data[current_fixation_start_index : i]
            if potential_fixation_window:
                potential_duration = potential_fixation_window[-1].get('timestamp', 0) - potential_fixation_window[0].get('timestamp', 0)
                if potential_duration >= duration_threshold_ms:
                    fixations.append(potential_fixation_window)
            current_fixation_start_index = i
        elif i == len(gaze_data) - 1 and duration >= duration_threshold_ms:
             fixations.append(window_points)

    num_fixations = len(fixations)
    avg_fixation_duration = np.mean([f[-1].get('timestamp', 0) - f[0].get('timestamp', 0) for f in fixations]) if fixations else 0

    if gaze_data:
        all_x = [p.get('x', 0) for p in gaze_data]
        all_y = [p.get('y', 0) for p in gaze_data]
        gaze_dispersion_x = np.std(all_x) if len(all_x) > 1 else 0
        gaze_dispersion_y = np.std(all_y) if len(all_y) > 1 else 0
    else:
        gaze_dispersion_x = 0
        gaze_dispersion_y = 0

    return {
        'num_fixations': num_fixations,
        'avg_fixation_duration': float(avg_fixation_duration),
        'gaze_dispersion_x': float(gaze_dispersion_x),
        'gaze_dispersion_y': float(gaze_dispersion_y)
    }

class CustomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10, random_state=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.max_features = max_features
        self.trees = []
        self.features_list = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.features_list = []
        n_features = X.shape[1]
        for i in range(self.n_estimators):
            if self.max_features == 'sqrt':
                n_sub_features = max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                n_sub_features = max(1, int(np.log2(n_features)))
            elif isinstance(self.max_features, int):
                n_sub_features = min(n_features, self.max_features)
            else:
                n_sub_features = n_features
            feature_indices = np.random.choice(n_features, n_sub_features, replace=False)
            tree_max_depth = self.max_depth + np.random.randint(-2, 3)
            tree_min_samples_split = max(2, self.min_samples_split + np.random.randint(-3, 4))
            tree_random_state = np.random.randint(0, 100000)
            tree = DecisionTreeRegressor(
                max_depth=tree_max_depth,
                min_samples_split=tree_min_samples_split,
                random_state=tree_random_state
            )
            idx = np.random.choice(len(X), size=int(len(X) * 0.8), replace=True)
            X_boot = X[idx][:, feature_indices]
            y_boot = y[idx]
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.features_list.append(feature_indices)
        return self

    def predict(self, X):
        preds = np.zeros((len(X), len(self.trees)))
        for i, tree in enumerate(self.trees):
            X_sub = X[:, self.features_list[i]]
            preds[:, i] = tree.predict(X_sub)
        return np.mean(preds, axis=1)

class ITEAnalyzer:
    def __init__(self, experiments_folder='data/experiments'):
        self.experiments_folder = experiments_folder

    def load_all_sessions(self):
        """Загружает все экспериментальные json-файлы в датафрейм"""
        records = []
        for fname in os.listdir(self.experiments_folder):
            if fname.endswith('.json'):
                with open(os.path.join(self.experiments_folder, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user_id = data.get('user_id', data.get('sessionId', fname))
                    browser = data.get('browser', {})
                    if isinstance(browser, dict):
                        browser_name = browser.get('name', 'Unknown')
                    else:
                        browser_name = browser or 'Unknown'
                    tasks = data.get('tasks', [])
                    completed_tasks = [t for t in tasks if t.get('completed')]
                    completion_rate = len(completed_tasks) / len(tasks) if tasks else 0
                    avg_task_duration = np.mean([t.get('duration', 0) for t in completed_tasks]) if completed_tasks else 0
                    total_gaze_points = len(data.get('gazeData', []))
                    total_emotions = len(data.get('emotionData', []))
                    total_interactions = len(data.get('interactionEvents', []))
                    positive_emotions = 0
                    negative_emotions = 0
                    neutral_emotions = 0
                    emotion_data = data.get('emotionData', [])
                    for emotion_record in emotion_data:
                        dominant_emotion = emotion_record.get('dominantEmotion')
                        if dominant_emotion == 'happy':
                            positive_emotions += 1
                        elif dominant_emotion in ['angry', 'sad', 'fearful', 'disgusted']:
                            negative_emotions += 1
                        elif dominant_emotion == 'neutral':
                            neutral_emotions += 1

                    emotion_transitions_count = 0
                    if emotion_data:
                        previous_emotion = emotion_data[0].get('dominantEmotion')
                        for i in range(1, len(emotion_data)):
                            current_emotion = emotion_data[i].get('dominantEmotion')
                            if current_emotion and previous_emotion and current_emotion != previous_emotion:
                                emotion_transitions_count += 1
                            previous_emotion = current_emotion

                    gaze_data = data.get('gazeData', [])
                    gaze_metrics = detect_fixations(gaze_data)

                    records.append({
                        'user_id': user_id,
                        'session_id': data.get('sessionId', fname),
                        'browser': browser_name,
                        'completion_rate': completion_rate,
                        'avg_task_duration': avg_task_duration,
                        'total_gaze_points': total_gaze_points,
                        'total_emotions': total_emotions, 
                        'total_interactions': total_interactions,
                        'positive_emotions': positive_emotions,
                        'negative_emotions': negative_emotions,
                        'neutral_emotions': neutral_emotions,
                        'emotion_transitions_count': emotion_transitions_count,
                        'gaze_num_fixations': gaze_metrics['num_fixations'],
                        'gaze_avg_fixation_duration': gaze_metrics['avg_fixation_duration'],
                        'gaze_dispersion_x': gaze_metrics['gaze_dispersion_x'],
                        'gaze_dispersion_y': gaze_metrics['gaze_dispersion_y'],
                    })
        return pd.DataFrame(records)

    def analyze_experiment(self, experiment_data=None):
        """
        Анализирует все реальные экспериментальные данные.
        Если experiment_data передан — добавляет его к анализу (например, для онлайн-анализа).
        """
        individual_ites_list = []
        try:
            df = self.load_all_sessions()
            if experiment_data:
                user_id = experiment_data.get('user_id', experiment_data.get('sessionId', 'current'))
                browser = experiment_data.get('browser', {})
                if isinstance(browser, dict):
                    browser_name = browser.get('name', 'Unknown')
                else:
                    browser_name = browser or 'Unknown'
                tasks = experiment_data.get('tasks', [])
                completed_tasks = [t for t in tasks if t.get('completed')]
                completion_rate = len(completed_tasks) / len(tasks) if tasks else 0
                avg_task_duration = np.mean([t.get('duration', 0) for t in completed_tasks]) if completed_tasks else 0
                total_gaze_points = len(experiment_data.get('gazeData', []))
                total_emotions = len(experiment_data.get('emotionData', []))
                total_interactions = len(experiment_data.get('interactionEvents', []))
                
                positive_emotions = 0
                negative_emotions = 0
                neutral_emotions = 0
                emotion_data = experiment_data.get('emotionData', [])
                for emotion_record in emotion_data:
                    dominant_emotion = emotion_record.get('dominantEmotion')
                    if dominant_emotion == 'happy':
                        positive_emotions += 1
                    elif dominant_emotion in ['angry', 'sad', 'fearful', 'disgusted']:
                        negative_emotions += 1
                    elif dominant_emotion == 'neutral':
                        neutral_emotions += 1
                
                emotion_transitions_count = 0
                if emotion_data:
                    previous_emotion = emotion_data[0].get('dominantEmotion')
                    for i in range(1, len(emotion_data)):
                        current_emotion = emotion_data[i].get('dominantEmotion')
                        if current_emotion and previous_emotion and current_emotion != previous_emotion:
                            emotion_transitions_count += 1
                        previous_emotion = current_emotion

                gaze_data = experiment_data.get('gazeData', [])
                gaze_metrics = detect_fixations(gaze_data)

                df_new = pd.DataFrame([{
                    'user_id': user_id,
                    'session_id': experiment_data.get('sessionId', 'current'),
                    'browser': browser_name,
                    'completion_rate': completion_rate,
                    'avg_task_duration': avg_task_duration,
                    'total_gaze_points': total_gaze_points,
                    'total_emotions': total_emotions, 
                    'total_interactions': total_interactions,
                    'positive_emotions': positive_emotions,
                    'negative_emotions': negative_emotions,
                    'neutral_emotions': neutral_emotions,
                    'emotion_transitions_count': emotion_transitions_count,
                    'gaze_num_fixations': gaze_metrics['num_fixations'],
                    'gaze_avg_fixation_duration': gaze_metrics['avg_fixation_duration'],
                    'gaze_dispersion_x': gaze_metrics['gaze_dispersion_x'],
                    'gaze_dispersion_y': gaze_metrics['gaze_dispersion_y'],
                }])
                df = pd.concat([df, df_new], ignore_index=True)

            df['treatment'] = (df['browser'] == 'Chrome').astype(int)
            outcome = 'completion_rate'
            features = ['avg_task_duration', 'total_gaze_points', 'total_emotions', 'total_interactions', 'positive_emotions', 'negative_emotions', 'neutral_emotions', 'emotion_transitions_count', 'gaze_num_fixations', 'gaze_avg_fixation_duration', 'gaze_dispersion_x', 'gaze_dispersion_y']

            warning = None
            if len(df) < 10 or df['treatment'].nunique() < 2:
                warning = 'Внимание: мало данных для корректного анализа. Результаты могут быть некорректны.'
                return {
                    'session_count': int(len(df)),
                    'ite_mean': {}, 
                    'ite_std': {},  
                    'visualizations': {},
                    'individual_ites': [], 
                    'note': warning
                }

            if df['treatment'].nunique() < 2:
                warning = 'Внимание: недостаточно разных браузеров (Chrome или не-Chrome) для анализа ITE.'
                return {
                    'session_count': int(len(df)),
                    'ite_mean': {}, 
                    'ite_std': {}, 
                    'visualizations': {},
                    'individual_ites': [],
                    'note': warning
                }

            X = df[features].fillna(0).values
            y = df[outcome].values
            treatment = df['treatment'].values

            if len(X_treat) == 0 or len(y_treat) == 0 or len(X_ctrl) == 0 or len(y_ctrl) == 0:
                plt.figure(figsize=(6, 2))
                plt.text(0.5, 0.5, 'Недостаточно данных для анализа', ha='center', va='center', fontsize=12)
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                plot_b64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                return {
                    'session_count': int(len(df)),
                    'ite_mean': {},
                    'ite_std': {},  
                    'visualizations': {
                        'ite_boxplot': plot_b64 
                    },
                    'individual_ites': [], 
                    'note': 'Внимание: нет данных хотя бы для одной из групп (Chrome или не-Chrome). Анализ невозможен.'
                }

            # --- T-learner + Random Forest ---
            rf_treat = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf_ctrl = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=43)
            rf_treat.fit(X_treat, y_treat)
            rf_ctrl.fit(X_ctrl, y_ctrl)
            ite_rf = rf_treat.predict(X) - rf_ctrl.predict(X)

            # --- X-learner + Random Forest ---
            mu1_ctrl_rf = rf_treat.predict(X_ctrl) 
            D0_rf = mu1_ctrl_rf - y_ctrl
            mu0_treat_rf = rf_ctrl.predict(X_treat)
            D1_rf = y_treat - mu0_treat_rf 

            rf_D0 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=44)
            rf_D1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=45)
            rf_D0.fit(X_ctrl, D0_rf)
            rf_D1.fit(X_treat, D1_rf)

            tau0_rf = rf_D0.predict(X)
            tau1_rf = rf_D1.predict(X)

            p = np.mean(treatment)

            ite_x_rf = (1 - p) * tau0_rf + p * tau1_rf

            # --- T-learner + Custom Forest ---
            cf_treat = CustomForest(n_estimators=100, max_depth=10, random_state=42)
            cf_ctrl = CustomForest(n_estimators=100, max_depth=10, random_state=43)
            cf_treat.fit(X_treat, y_treat)
            cf_ctrl.fit(X_ctrl, y_ctrl)
            ite_cf = cf_treat.predict(X) - cf_ctrl.predict(X)

            # --- X-learner + Custom Forest ---
            mu1_ctrl_cf = cf_treat.predict(X_ctrl)
            D0_cf = mu1_ctrl_cf - y_ctrl 
            mu0_treat_cf = cf_ctrl.predict(X_treat) 
            D1_cf = y_treat - mu0_treat_cf 

            cf_D0 = CustomForest(n_estimators=100, max_depth=10, random_state=44)
            cf_D1 = CustomForest(n_estimators=100, max_depth=10, random_state=45)
            cf_D0.fit(X_ctrl, D0_cf)
            cf_D1.fit(X_treat, D1_cf)

            tau0_cf = cf_D0.predict(X)
            tau1_cf = cf_D1.predict(X)

            ite_x_cf = (1 - p) * tau0_cf + p * tau1_cf

            # --- Визуализация: boxplot для всех моделей ---
            plt.figure(figsize=(10, 6))
            plt.boxplot([
                ite_rf, ite_x_rf, ite_cf, ite_x_cf
            ], labels=[
                'T-learner + RF', 'X-learner + RF', 'T-learner + CF', 'X-learner + CF'
            ], patch_artist=True)
            plt.title('Сравнение ITE для всех моделей')
            plt.ylabel('ITE')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            boxplot_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # --- Визуализация: Scatter plot индивидуальных ITE (например, для T-learner + RF) ---
            plt.figure(figsize=(12, 6))
            # Используем уникальные user_id или просто индекс как x для наглядности
            # Цветом обозначим браузер
            colors = df['browser'].astype('category').cat.codes
            scatter = plt.scatter(df.index, ite_rf, c=colors, cmap='viridis', alpha=0.6)
            plt.title('Индивидуальные оценки ITE (T-learner + RF) по сессиям')
            plt.xlabel('Индекс сессии в данных') # Или можно использовать user_id, если сессий у юзера мало
            plt.ylabel('ITE (T-learner + RF)')
            # Добавляем легенду для цветов браузеров
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                                        markerfacecolor=scatter.cmap(scatter.norm(code)), markersize=10)
                               for code, cat in enumerate(df['browser'].astype('category').cat.categories)]
            plt.legend(handles=legend_elements, title='Браузер')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            scatter_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # --- Сбор индивидуальных ITE по сессиям ---
            individual_ites_list = []
            anxiety_indices = []
            for i in range(len(df)):
                A_disp = np.sqrt(df.iloc[i]['gaze_dispersion_x'] ** 2 + df.iloc[i]['gaze_dispersion_y'] ** 2)
                t_fix_avg = df.iloc[i]['gaze_avg_fixation_duration']
                f_fix = df.iloc[i]['gaze_num_fixations'] / (df.iloc[i]['avg_task_duration'] * len(df) if df.iloc[i]['avg_task_duration'] > 0 else 1)
                anxiety_indices.append({
                    'user_id': df.iloc[i]['user_id'],
                    'session_id': df.iloc[i]['session_id'],
                    'browser': df.iloc[i]['browser'],
                    'A_disp': A_disp,
                    't_fix_avg': t_fix_avg,
                    'f_fix': f_fix
                })
                individual_ites_list.append({
                    'user_id': df.iloc[i]['user_id'],
                    'session_id': df.iloc[i]['session_id'],
                    'browser': df.iloc[i]['browser'],
                    'T-learner + RF': float(ite_rf[i]),
                    'X-learner + RF': float(ite_x_rf[i]),
                    'T-learner + CF': float(ite_cf[i]),
                    'X-learner + CF': float(ite_x_cf[i]),
                })

            A_disp_ref = np.mean([a['A_disp'] for a in anxiety_indices])
            t_fix_avg_ref = np.mean([a['t_fix_avg'] for a in anxiety_indices])
            f_fix_ref = np.mean([a['f_fix'] for a in anxiety_indices])
            w1 = w2 = w3 = 1.0
            for a in anxiety_indices:
                a['AnxietyIndex'] = (
                    w1 * (a['A_disp'] / A_disp_ref if A_disp_ref else 0) +
                    w2 * (a['f_fix'] / f_fix_ref if f_fix_ref else 0) -
                    w3 * (a['t_fix_avg'] / t_fix_avg_ref if t_fix_avg_ref else 0)
                )

            plt.figure(figsize=(12, 6))
            colors = pd.Series([a['browser'] for a in anxiety_indices]).astype('category').cat.codes
            anxiety_vals = [a['AnxietyIndex'] for a in anxiety_indices]
            scatter = plt.scatter(range(len(anxiety_indices)), anxiety_vals, c=colors, cmap='viridis', alpha=0.6)
            plt.title('Индекс тревожности по сессиям (gaze)')
            plt.xlabel('Индекс сессии в данных')
            plt.ylabel('AnxietyIndex')
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                                          markerfacecolor=scatter.cmap(scatter.norm(code)), markersize=10)
                               for code, cat in enumerate(pd.Series([a['browser'] for a in anxiety_indices]).astype('category').cat.categories)]
            plt.legend(handles=legend_elements, title='Браузер')
            plt.grid(True, linestyle='--', alpha=0.6)
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            anxiety_scatter_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # Сохраняем таблицу всех признаков по сессиям
            features_to_save = [
                'participantNumber', 'user_id', 'session_id', 'browser',
                'avg_task_duration', 'total_gaze_points', 'total_emotions', 'total_interactions',
                'positive_emotions', 'negative_emotions', 'neutral_emotions', 'emotion_transitions_count',
                'gaze_num_fixations', 'gaze_avg_fixation_duration', 'gaze_dispersion_x', 'gaze_dispersion_y'
            ]
            # Если participantNumber нет в df, добавим пустой столбец
            if 'participantNumber' not in df.columns:
                df['participantNumber'] = ''
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            features_csv = f"data/results/all_features_table_{timestamp_str}.csv"
            try:
                df[features_to_save].to_csv(features_csv, index=False)
            except Exception as e:
                print(f"Ошибка при сохранении all_features_table: {e}")

            return {
                'session_count': int(len(df)),
                'ite_mean': {
                    'T-learner + RF': float(np.mean(ite_rf)),
                    'X-learner + RF': float(np.mean(ite_x_rf)),
                    'T-learner + CF': float(np.mean(ite_cf)),
                    'X-learner + CF': float(np.mean(ite_x_cf)),
                },
                'ite_std': {
                    'T-learner + RF': float(np.std(ite_rf)),
                    'X-learner + RF': float(np.std(ite_x_rf)),
                    'T-learner + CF': float(np.std(ite_cf)),
                    'X-learner + CF': float(np.std(ite_x_cf)),
                },
                'visualizations': {
                    'ite_boxplot': boxplot_b64,
                    'ite_scatter': scatter_b64,
                    'anxiety_scatter': anxiety_scatter_b64
                },
                'individual_ites': individual_ites_list,
                'anxiety_indices': anxiety_indices,
                'note': warning or 'Анализ выполнен с помощью авторских реализаций T-learner/X-learner + Random Forest/Custom Forest.'
            }
        except Exception as e:
            return {
                'session_count': int(len(df)) if 'df' in locals() else 0,
                'ite_mean': {}, 
                'ite_std': {}, 
                'visualizations': {},
                'individual_ites': [],
                'error': str(e),
                'note': 'Произошла ошибка в процессе анализа.'
            }

    def analyze_experiment_list(self, experiment_list):
        """
        Анализирует список json-объектов (экспериментов), как если бы они были считаны из папки.
        """
        # Собираем все записи в датафрейм
        records = []
        for data in experiment_list:
            user_id = data.get('user_id', data.get('sessionId', 'unknown'))
            browser = data.get('browser', {})
            if isinstance(browser, dict):
                browser_name = browser.get('name', 'Unknown')
            else:
                browser_name = browser or 'Unknown'
            tasks = data.get('tasks', [])
            completed_tasks = [t for t in tasks if t.get('completed')]
            completion_rate = len(completed_tasks) / len(tasks) if tasks else 0
            avg_task_duration = np.mean([t.get('duration', 0) for t in completed_tasks]) if completed_tasks else 0
            total_gaze_points = len(data.get('gazeData', []))
            total_emotions = len(data.get('emotionData', []))
            total_interactions = len(data.get('interactionEvents', []))
            positive_emotions = 0
            negative_emotions = 0
            neutral_emotions = 0
            emotion_data = data.get('emotionData', [])
            for emotion_record in emotion_data:
                dominant_emotion = emotion_record.get('dominantEmotion')
                if dominant_emotion == 'happy':
                    positive_emotions += 1
                elif dominant_emotion in ['angry', 'sad', 'fearful', 'disgusted']:
                    negative_emotions += 1
                elif dominant_emotion == 'neutral':
                    neutral_emotions += 1

            emotion_transitions_count = 0
            if emotion_data:
                previous_emotion = emotion_data[0].get('dominantEmotion')
                for i in range(1, len(emotion_data)):
                    current_emotion = emotion_data[i].get('dominantEmotion')
                    if current_emotion and previous_emotion and current_emotion != previous_emotion:
                        emotion_transitions_count += 1
                    previous_emotion = current_emotion

            gaze_data = data.get('gazeData', [])
            gaze_metrics = detect_fixations(gaze_data)

            participant_number = data.get('participantNumber', '')

            records.append({
                'participantNumber': participant_number,
                'user_id': user_id,
                'session_id': data.get('sessionId', 'unknown'),
                'browser': browser_name,
                'completion_rate': completion_rate,
                'avg_task_duration': avg_task_duration,
                'total_gaze_points': total_gaze_points,
                'total_emotions': total_emotions,
                'total_interactions': total_interactions,
                'positive_emotions': positive_emotions,
                'negative_emotions': negative_emotions,
                'neutral_emotions': neutral_emotions,
                'emotion_transitions_count': emotion_transitions_count,
                'gaze_num_fixations': gaze_metrics['num_fixations'],
                'gaze_avg_fixation_duration': gaze_metrics['avg_fixation_duration'],
                'gaze_dispersion_x': gaze_metrics['gaze_dispersion_x'],
                'gaze_dispersion_y': gaze_metrics['gaze_dispersion_y'],
            })
        df = pd.DataFrame(records)
        # Если датафрейм пустой, нет данных для анализа
        if df.empty:
            return {
                'session_count': 0,
                'ite_mean': {}, 
                'ite_std': {},  
                'visualizations': {},
                'individual_ites': [], 
                'note': 'Нет данных для анализа.'
            }
        # Иначе, передаем датафрейм в analyze_experiment_from_df для проведения анализа
        return self.analyze_experiment_from_df(df)

    def analyze_experiment_from_df(self, df):
        def run_ite_analysis(df, features, outcome, treatment, label_suffix):
            X = df[features].fillna(0).values
            y = df[outcome].values
            X_treat = X[treatment == 1]
            y_treat = y[treatment == 1]
            X_ctrl = X[treatment == 0]
            y_ctrl = y[treatment == 0]
            if len(X_treat) == 0 or len(y_treat) == 0 or len(X_ctrl) == 0 or len(y_ctrl) == 0:
                return None
            # --- T-learner + Random Forest ---
            rf_treat = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            rf_ctrl = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=43)
            rf_treat.fit(X_treat, y_treat)
            rf_ctrl.fit(X_ctrl, y_ctrl)
            ite_rf = rf_treat.predict(X) - rf_ctrl.predict(X)
            # --- X-learner + Random Forest ---
            mu1_ctrl_rf = rf_treat.predict(X_ctrl)
            D0_rf = mu1_ctrl_rf - y_ctrl
            mu0_treat_rf = rf_ctrl.predict(X_treat)
            D1_rf = y_treat - mu0_treat_rf
            rf_D0 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=44)
            rf_D1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=45)
            rf_D0.fit(X_ctrl, D0_rf)
            rf_D1.fit(X_treat, D1_rf)
            tau0_rf = rf_D0.predict(X)
            tau1_rf = rf_D1.predict(X)
            p = np.mean(treatment)
            ite_x_rf = (1 - p) * tau0_rf + p * tau1_rf
            # --- T-learner + Custom Forest ---
            cf_treat = CustomForest(n_estimators=100, max_depth=10, random_state=42)
            cf_ctrl = CustomForest(n_estimators=100, max_depth=10, random_state=43)
            cf_treat.fit(X_treat, y_treat)
            cf_ctrl.fit(X_ctrl, y_ctrl)
            ite_cf = cf_treat.predict(X) - cf_ctrl.predict(X)
            # --- X-learner + Custom Forest ---
            mu1_ctrl_cf = cf_treat.predict(X_ctrl)
            D0_cf = mu1_ctrl_cf - y_ctrl
            mu0_treat_cf = cf_ctrl.predict(X_treat)
            D1_cf = y_treat - mu0_treat_cf
            cf_D0 = CustomForest(n_estimators=100, max_depth=10, random_state=44)
            cf_D1 = CustomForest(n_estimators=100, max_depth=10, random_state=45)
            cf_D0.fit(X_ctrl, D0_cf)
            cf_D1.fit(X_treat, D1_cf)
            tau0_cf = cf_D0.predict(X)
            tau1_cf = cf_D1.predict(X)
            ite_x_cf = (1 - p) * tau0_cf + p * tau1_cf
            # --- Индивидуальные эффекты по респондентам ---
            individual_ites = []
            for i in range(len(df)):
                individual_ites.append({
                    'participantNumber': df.iloc[i].get('participantNumber', ''),
                    'user_id': df.iloc[i].get('user_id', ''),
                    'session_id': df.iloc[i].get('session_id', ''),
                    'browser': df.iloc[i].get('browser', ''),
                    f'T-learner + RF{label_suffix}': float(ite_rf[i]),
                    f'X-learner + RF{label_suffix}': float(ite_x_rf[i]),
                    f'T-learner + CF{label_suffix}': float(ite_cf[i]),
                    f'X-learner + CF{label_suffix}': float(ite_x_cf[i]),
                })
            # --- Визуализация: boxplot для всех моделей ---
            plt.figure(figsize=(10, 6))
            plt.boxplot([
                ite_rf, ite_x_rf, ite_cf, ite_x_cf
            ], labels=[
                f'T-learner + RF{label_suffix}', f'X-learner + RF{label_suffix}', f'T-learner + CF{label_suffix}', f'X-learner + CF{label_suffix}'
            ], patch_artist=True)
            plt.title(f'Сравнение ITE для всех моделей {label_suffix}')
            plt.ylabel('ITE')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            boxplot_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return {
                'ite_mean': {
                    f'T-learner + RF{label_suffix}': float(np.mean(ite_rf)),
                    f'X-learner + RF{label_suffix}': float(np.mean(ite_x_rf)),
                    f'T-learner + CF{label_suffix}': float(np.mean(ite_cf)),
                    f'X-learner + CF{label_suffix}': float(np.mean(ite_x_cf)),
                },
                'ite_std': {
                    f'T-learner + RF{label_suffix}': float(np.std(ite_rf)),
                    f'X-learner + RF{label_suffix}': float(np.std(ite_x_rf)),
                    f'T-learner + CF{label_suffix}': float(np.std(ite_cf)),
                    f'X-learner + CF{label_suffix}': float(np.std(ite_x_cf)),
                },
                'visualizations': {
                    f'ite_boxplot{label_suffix}': boxplot_b64
                },
                'individual_ites': individual_ites
            }

        df['treatment'] = (df['browser'] == 'Chrome').astype(int)
        outcome = 'completion_rate'
        # 1. Полный интегрированный вектор X
        features_full = ['avg_task_duration', 'total_gaze_points', 'total_emotions', 'total_interactions',
                         'positive_emotions', 'negative_emotions', 'neutral_emotions', 'emotion_transitions_count',
                         'gaze_num_fixations', 'gaze_avg_fixation_duration', 'gaze_dispersion_x', 'gaze_dispersion_y']
        # 2. Без gaze-метрик
        features_no_gaze = ['avg_task_duration', 'total_emotions', 'total_interactions',
                            'positive_emotions', 'negative_emotions', 'neutral_emotions', 'emotion_transitions_count']
        # 3. Без emotion-метрик
        features_no_emotion = ['avg_task_duration', 'total_gaze_points', 'total_interactions',
                               'gaze_num_fixations', 'gaze_avg_fixation_duration', 'gaze_dispersion_x', 'gaze_dispersion_y']
        treatment = df['treatment'].values
        results = {}
        results['full_features'] = run_ite_analysis(df, features_full, outcome, treatment, ' (integrated)')
        results['no_gaze'] = run_ite_analysis(df, features_no_gaze, outcome, treatment, ' (no gaze)')
        results['no_emotion'] = run_ite_analysis(df, features_no_emotion, outcome, treatment, ' (no emotion)')
        results['session_count'] = int(len(df))
        results['note'] = 'Comparison of three feature vector variants: integrated, no gaze, no emotion.'
        # Сохраняем таблицу всех признаков по сессиям
        features_to_save = [
            'participantNumber', 'user_id', 'session_id', 'browser',
            'avg_task_duration', 'total_gaze_points', 'total_emotions', 'total_interactions',
            'positive_emotions', 'negative_emotions', 'neutral_emotions', 'emotion_transitions_count',
            'gaze_num_fixations', 'gaze_avg_fixation_duration', 'gaze_dispersion_x', 'gaze_dispersion_y'
        ]
        # Если participantNumber нет в df, добавим пустой столбец
        if 'participantNumber' not in df.columns:
            df['participantNumber'] = ''
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_csv = f"data/results/all_features_table_{timestamp_str}.csv"
        try:
            df[features_to_save].to_csv(features_csv, index=False)
        except Exception as e:
            print(f"Ошибка при сохранении all_features_table: {e}")
        return results 