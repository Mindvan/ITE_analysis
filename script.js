function generateSessionId() {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function getBrowserName() {
  if (typeof navigator === 'undefined') return 'Unknown';
  const userAgent = navigator.userAgent;
  if (userAgent.includes('Firefox')) return 'Firefox';
  if (userAgent.includes('Chrome')) return 'Chrome';
  if (userAgent.includes('Safari')) return 'Safari';
  if (userAgent.includes('Edge')) return 'Edge';
  return 'Unknown';
}

function getBrowserVersion() {
  if (typeof navigator === 'undefined') return 'Unknown';
  const userAgent = navigator.userAgent;
  const match = userAgent.match(/(Firefox|Chrome|Safari|Edge)\/(\d+)/);
  return match ? match[2] : 'Unknown';
}

let experimentData = {
  sessionId: generateSessionId(),
  timestamp: Date.now(),
  startTime: null,
  endTime: null,
  duration: null,
  
  browser: {
    name: getBrowserName(),
    version: getBrowserVersion(),
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language
  },
  
  screen: {
    width: screen.width,
    height: screen.height,
    availWidth: screen.availWidth,
    availHeight: screen.availHeight,
    pixelRatio: window.devicePixelRatio,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight
    }
  },
  
  tasks: [],
  currentTaskId: null,
  gazeData: [],
  emotionData: [],
  interactionEvents: [],
  errors: []
}

let timerInterval = null;
let isExperimentRunning = false;

if (typeof document !== 'undefined') {
  const startBtn = document.getElementById('startBtn')
  const stopBtn = document.getElementById('stopBtn')
  const tasksBtn = document.getElementById('tasksBtn')
  const timerDisplay = document.getElementById('timer')
  const modal = document.getElementById('tasksModal')
  const closeBtn = document.querySelector('.close')


  const controlsDiv = document.querySelector('.controls');


  const experimentBtn = document.getElementById('experimentBtn');

  if (tasksBtn) tasksBtn.disabled = true;

  experimentBtn.addEventListener('click', () => {
    if (!isExperimentRunning) {
      isExperimentRunning = true;
      const oldExperimentData = { ...experimentData };  // Сохраняем старые данные
      experimentData = {
        startTime: Date.now(),
        endTime: null,
        duration: null,
        gazeData: [],
        emotionData: [],
        sessionId: generateSessionId(),
        timestamp: Date.now(),
        browser: oldExperimentData.browser || {  // Используем старые данные о браузере
          name: getBrowserName(),
          version: getBrowserVersion(),
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          language: navigator.language
        },
        screen: oldExperimentData.screen || {  // Используем старые данные об экране
          width: screen.width,
          height: screen.height,
          availWidth: screen.availWidth,
          availHeight: screen.availHeight,
          pixelRatio: window.devicePixelRatio,
          viewport: {
            width: window.innerWidth,
            height: window.innerHeight
          }
        },
        tasks: [],
        currentTaskId: null,
        interactionEvents: [],
        errors: []
      };
      const info = localStorage.getItem('participantInfo');
      if (info) {
        try {
          const parsed = JSON.parse(info);
          experimentData.participantNumber = parsed.participantNumber;
          experimentData.glasses = parsed.glasses;
          experimentData.consentData = parsed.consentData;
          experimentData.consentExperiment = parsed.consentExperiment;
        } catch (e) {

        }
      }
      experimentBtn.textContent = 'Завершить';
      experimentBtn.classList.add('btn-stop');
      experimentBtn.classList.remove('btn-start');
      timerInterval = setInterval(updateTimer, 1000);
      updateTimer();
      if (tasksBtn) tasksBtn.disabled = false;
    } else {

      isExperimentRunning = false;
      experimentData.endTime = Date.now();
      experimentData.duration = experimentData.endTime - experimentData.startTime;
      experimentBtn.textContent = 'Начать';
      experimentBtn.classList.add('btn-start');
      experimentBtn.classList.remove('btn-stop');
      clearInterval(timerInterval);
      saveExperimentData();
      if (tasksBtn) tasksBtn.disabled = true;
    }
  });


  function formatTime(ms) {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
  }


  function updateTimer() {
    const currentTime = Date.now()
    const elapsedTime = currentTime - experimentData.startTime
    timerDisplay.textContent = `Время: ${formatTime(elapsedTime)}`
  }


  async function saveExperimentData() {
    if (experimentData.currentTaskId) {
      completeTask(experimentData.currentTaskId)
    }


    if (typeof window !== 'undefined' && typeof window.calibrationAccuracy !== 'undefined') {
      experimentData.calibrationAccuracy = window.calibrationAccuracy;
    } else {
      experimentData.calibrationAccuracy = null;
    }


    if (!experimentData.browser) {
      experimentData.browser = {
        name: getBrowserName(),
        version: getBrowserVersion(),
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language
      };
    }

    experimentData.statistics = {
      totalTasks: experimentData.tasks.length,
      completedTasks: experimentData.tasks.filter(t => t.completed).length,
      totalGazePoints: experimentData.gazeData.length,
      totalEmotionSamples: experimentData.emotionData.length,
      totalInteractions: experimentData.interactionEvents.length,
      avgTaskDuration: calculateAverageTaskDuration(),
      emotionDistribution: calculateEmotionDistribution()
    }
    
    
    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ experimentData })
      })
      const result = await response.json()
      if (result.status === 'success') {
        console.log('Данные отправлены на сервер:', result)
        

        const analyzeTwoSessionsBtn = document.createElement('button');
        analyzeTwoSessionsBtn.textContent = 'Анализировать две сессии';
        analyzeTwoSessionsBtn.className = 'btn btn-info';
        analyzeTwoSessionsBtn.style.marginTop = '10px';
        analyzeTwoSessionsBtn.style.marginLeft = '10px';
        analyzeTwoSessionsBtn.onclick = () => {
          const sessionIds = prompt('Введите ID двух сессий через запятую (например: session_123, session_456):');
          if (sessionIds) {
            const ids = sessionIds.split(',').map(id => id.trim());
            if (ids.length === 2) {
              analyzeSessions(ids);
            } else {
              alert('Пожалуйста, введите ровно два ID сессий через запятую');
            }
          }
        };
        

        const experimentBtn = document.getElementById('experimentBtn');
        experimentBtn.parentNode.insertBefore(analyzeTwoSessionsBtn, experimentBtn.nextSibling);
        
        alert('Данные эксперимента сохранены! Для анализа необходимо провести эксперимент хотя бы в двух разных браузерах (Chrome и не-Chrome). Когда будет достаточно данных, нажмите кнопку \'Анализировать две сессии\'.');
      } else {
        console.error('Ошибка отправки:', result.error)
      }
    } catch (error) {
      console.error('Ошибка соединения с сервером:', error)
    }
  }


  function showLoaderModal(message = 'Выполняется анализ, пожалуйста, подождите...') {
    let loaderModal = document.getElementById('loaderModal');
    if (!loaderModal) {
      loaderModal = document.createElement('div');
      loaderModal.id = 'loaderModal';
      loaderModal.style.position = 'fixed';
      loaderModal.style.top = 0;
      loaderModal.style.left = 0;
      loaderModal.style.width = '100vw';
      loaderModal.style.height = '100vh';
      loaderModal.style.background = 'rgba(0,0,0,0.4)';
      loaderModal.style.display = 'flex';
      loaderModal.style.alignItems = 'center';
      loaderModal.style.justifyContent = 'center';
      loaderModal.style.zIndex = 9999;
      loaderModal.innerHTML = `
        <div style="background:#fff;padding:32px 48px;border-radius:12px;box-shadow:0 2px 16px #0002;text-align:center;min-width:320px;">
          <div class="loader" style="margin-bottom:16px;width:48px;height:48px;border:6px solid #eee;border-top:6px solid #2196f3;border-radius:50%;animation:spin 1s linear infinite;"></div>
          <div style="font-size:18px;">${message}</div>
        </div>
        <style>@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>
      `;
      document.body.appendChild(loaderModal);
    } else {
      loaderModal.style.display = 'flex';
    }
  }
  function hideLoaderModal() {
    const loaderModal = document.getElementById('loaderModal');
    if (loaderModal) loaderModal.style.display = 'none';
  }


  function showResultsModal(html) {

    const oldModal = document.getElementById('resultsModal');
    if (oldModal) oldModal.remove();

    const modal = document.createElement('div');
    modal.id = 'resultsModal';
    modal.className = 'custom-modal';
    modal.innerHTML = `
      <div class="custom-modal-content" style="width: 98vw; font-size: 14px; padding: 20px 24px 16px 24px; box-sizing: border-box;">
        <span class="close" onclick="this.parentElement.parentElement.remove()" style="font-size: 24px; float: right; cursor: pointer;">&times;</span>
        ${html}
      </div>
      <style>
        .custom-modal {
          display: block !important;
          position: fixed;
          z-index: 20000;
          left: 0; top: 0; width: 100vw; height: 100vh;
          background: rgba(0,0,0,0.25);
          overflow: auto;
        }
        .custom-modal-content {
          margin: 40px auto;
          background: #fff;
          border-radius: 10px;
          box-shadow: 0 4px 32px #0002;
        }
        @media (max-width: 750px) {
          .custom-modal-content { max-width: 98vw; }
        }
      </style>
    `;
    document.body.appendChild(modal);
  }


  async function analyzeSessions(sessionIds) {
    try {
        showLoaderModal('Выполняется анализ двух сессий...');
        
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sessionIds })
        });

        const result = await response.json();
        hideLoaderModal();
        
        if (result.error) {
            alert(`Ошибка анализа: ${result.error}\n${result.note || ''}`);
            return;
        }


        let effectDirection = '';
        if (result.individual_effect.value > 0) {
            effectDirection = '<div style="margin:10px 0 0 0; color: #388e3c; font-weight: bold;">Эффект в пользу Chrome</div>';
        } else if (result.individual_effect.value < 0) {
            effectDirection = '<div style="margin:10px 0 0 0; color: #1976d2; font-weight: bold;">Эффект в пользу Firefox</div>';
        } else {
            effectDirection = '<div style="margin:10px 0 0 0; color: #888; font-weight: bold;">Эффект отсутствует</div>';
        }

        function getEmotionPercents(emotionDistribution) {
            const allEmotions = ['happy','sad','angry','surprised','disgusted','fearful','neutral'];
            const total = allEmotions.reduce((sum, e) => sum + (emotionDistribution[e] || 0), 0) || 1;
            const percents = {};
            allEmotions.forEach(e => {
                percents[e] = ((emotionDistribution[e] || 0) / total * 100).toFixed(1);
            });
            return percents;
        }
        const chromePercents = getEmotionPercents(result.emotion_stats.Chrome.distribution);
        const firefoxPercents = getEmotionPercents(result.emotion_stats.Firefox.distribution);

        let emotionTable = `<table style="margin-top:10px; border-collapse:collapse; width:100%; font-size:13px;">
            <tr><th style='text-align:left;'>Эмоция</th><th>Chrome (%)</th><th>Firefox (%)</th></tr>`;
        ['happy','sad','angry','surprised','disgusted','fearful','neutral'].forEach(emotion => {
            emotionTable += `<tr><td>${emotion}</td><td>${chromePercents[emotion]}</td><td>${firefoxPercents[emotion]}</td></tr>`;
        });
        emotionTable += '</table>';

        const chromeIcon = `<img src="assets/logo_chrome.png" alt="Chrome" style="height:22px;vertical-align:middle;margin-right:6px;">`;
        const firefoxIcon = `<img src="assets/logo_firefox.png" alt="Firefox" style="height:22px;vertical-align:middle;margin-right:6px;">`;

        let variantsTable = '';
        if (result.individual_effects_variants) {
            variantsTable = `<div style='margin:18px 0 0 0;'>
                <b style='font-size:15px;'>Варианты индивидуального эффекта:</b>
                <table style='margin-top:8px; border-collapse:collapse; width:100%; font-size:13px;'>
                    <tr><th style='text-align:left;'>Вариант</th><th>Процент</th><th>ITE</th></tr>
                    <tr><td>Интегрированный (все метрики)</td><td>${(result.individual_effects_variants.integrated * 100).toFixed(1)}%</td><td>${result.individual_effects_variants.integrated.toFixed(3)}</td></tr>
                    <tr><td>Только эмоции (без gaze)</td><td>${(result.individual_effects_variants.no_gaze * 100).toFixed(1)}%</td><td>${result.individual_effects_variants.no_gaze.toFixed(3)}</td></tr>
                    <tr><td>Только gaze (без эмоций)</td><td>${(result.individual_effects_variants.no_emotion * 100).toFixed(1)}%</td><td>${result.individual_effects_variants.no_emotion.toFixed(3)}</td></tr>
                </table>
            </div>`;
        }

        let calibrationChrome = result.browser_comparison.Chrome.calibrationAccuracy !== undefined ? result.browser_comparison.Chrome.calibrationAccuracy : null;
        let calibrationFirefox = result.browser_comparison.Firefox.calibrationAccuracy !== undefined ? result.browser_comparison.Firefox.calibrationAccuracy : null;
        let anxietyChrome = result.browser_comparison.Chrome.anxietyIndex !== undefined ? result.browser_comparison.Chrome.anxietyIndex : null;
        let anxietyFirefox = result.browser_comparison.Firefox.anxietyIndex !== undefined ? result.browser_comparison.Firefox.anxietyIndex : null;

        let ciLow = result.individual_effect.confidence_interval[0];
        let ciHigh = result.individual_effect.confidence_interval[1];
        let ciWidth = Math.abs(ciHigh - ciLow);
        let ciText = '';
        if (ciLow < 0 && ciHigh < 0) {
            ciText = 'Эффект устойчив и статистически значим.';
        } else if (ciLow > 0 && ciHigh > 0) {
            ciText = 'Эффект устойчив и статистически значим.';
        } else if (ciLow < 0 && ciHigh > 0) {
            ciText = 'Разница между браузерами не доказана статистически.';
        }
        if (ciWidth > 0.3) {
            ciText += ' Оценка эффекта неуверенная (широкий интервал).';
        } else if (ciWidth < 0.15) {
            ciText += ' Оценка эффекта уверенная (узкий интервал).';
        }
        const html = `
            <h3 style="font-size: 22px; margin-bottom: 18px; color: #222; letter-spacing: 0.5px;">Результаты оценки пользовательского опыта по трекингу взгляда и мимике</h3>
            <div class="result-section" style="background: #f8fafc; box-shadow: 0 2px 16px #0001; border-radius: 12px; padding: 20px 24px 18px 24px; margin-bottom: 18px;">
                <h4 style="font-size: 17px; color: #333; margin-bottom: 10px;">Индивидуальный эффект воздействия</h4>
                <p style="font-size: 15px; margin: 0 0 6px 0;">Значение: <b>${(result.individual_effect.value * 100).toFixed(1)}%</b> <span style='color:#888;font-size:13px;'>(ITE: ${result.individual_effect.value.toFixed(3)})</span></p>
                <p style="font-size: 14px; margin: 0 0 6px 0; color: #666;">Доверительный интервал (95%): <b>[${(result.individual_effect.confidence_interval[0] * 100).toFixed(1)}%, ${(result.individual_effect.confidence_interval[1] * 100).toFixed(1)}%]</b></p>
                ${effectDirection}
                ${variantsTable}
                <p style="font-size: 13px; color: #1976d2; margin: 10px 0 0 0;">${ciText}</p>
            </div>
            <div class="result-section" style="background: #fff; box-shadow: 0 2px 16px #0001; border-radius: 12px; padding: 20px 24px 18px 24px;">
                <h4 style="font-size: 17px; color: #333; margin-bottom: 10px;">Сравнение браузеров и эмоций</h4>
                <div class="browser-comparison" style="display: flex; justify-content: space-around; gap: 24px; margin-bottom: 18px;">
                    <div class="browser-stats" style="background: linear-gradient(90deg,#e3f2fd,#fff); border-radius: 8px; padding: 14px 18px; min-width: 220px; box-shadow: 0 1px 6px #90caf940;">
                        <h5 style="color: #1976d2; font-size: 16px; margin-bottom: 8px;">${chromeIcon}Chrome</h5>
                        <p>Среднее время выполнения: <b>${result.browser_comparison.Chrome.avg_task_duration.toFixed(2)} сек</b></p>
                        <p>Калибровка (точность): <b>${calibrationChrome !== null ? calibrationChrome.toFixed(1) + '%' : 'Н/Д'}</b></p>
                        <p>Индекс тревожности (отвлечённости): <b>${anxietyChrome !== null ? anxietyChrome.toFixed(3) : 'Н/Д'}</b></p>
                        <div style="margin-top: 10px; font-size: 13px; color: #666;">
                            <b>Эмоции:</b><br>
                            Положительные: ${result.emotion_stats.Chrome.positive}<br>
                            Отрицательные: ${result.emotion_stats.Chrome.negative}<br>
                            Нейтральные: ${result.emotion_stats.Chrome.neutral}
                        </div>
                    </div>
                    <div class="browser-stats" style="background: linear-gradient(90deg,#fff3e0,#fff); border-radius: 8px; padding: 14px 18px; min-width: 220px; box-shadow: 0 1px 6px #ffcc8040;">
                        <h5 style="color: #ff9800; font-size: 16px; margin-bottom: 8px;">${firefoxIcon}Firefox</h5>
                        <p>Среднее время выполнения: <b>${result.browser_comparison.Firefox.avg_task_duration.toFixed(2)} сек</b></p>
                        <p>Калибровка (точность): <b>${calibrationFirefox !== null ? calibrationFirefox.toFixed(1) + '%' : 'Н/Д'}</b></p>
                        <p>Индекс тревожности (отвлечённости): <b>${anxietyFirefox !== null ? anxietyFirefox.toFixed(3) : 'Н/Д'}</b></p>
                        <div style="margin-top: 10px; font-size: 13px; color: #666;">
                            <b>Эмоции:</b><br>
                            Положительные: ${result.emotion_stats.Firefox.positive}<br>
                            Отрицательные: ${result.emotion_stats.Firefox.negative}<br>
                            Нейтральные: ${result.emotion_stats.Firefox.neutral}
                        </div>
                    </div>
                </div>
                <div style='margin-top:10px; font-size:12px; color:#888;'>
                    <b>Примечание:</b> Индекс тревожности (отвлечённости) — интегральный безразмерный показатель на основе трекинга взгляда, используется только для сравнения между сессиями. Дальнейшую интерпретацию должны дать психологи.
                </div>
                <div style='margin-top:16px;'>
                    <b style="font-size:15px;">Проценты всех эмоций:</b>
                    ${emotionTable}
                </div>
            </div>
        `;
        showResultsModal(html);

    } catch (error) {
        hideLoaderModal();
        console.error('Ошибка при анализе:', error);
        alert('Произошла ошибка при анализе данных');
    }
  }


  function addInteractionEvent(type, data = {}) {
    if (!isExperimentRunning) return
    
    experimentData.interactionEvents.push({
      timestamp: Date.now(),
      type: type,
      taskId: experimentData.currentTaskId,
      data: data
    })
  }


  function startTask(taskId, taskDescription) {
    const task = {
      id: taskId,
      description: taskDescription,
      startTime: Date.now(),
      endTime: null,
      duration: null,
      completed: false
    }
    
    experimentData.tasks.push(task)
    experimentData.currentTaskId = taskId
    
    
    addInteractionEvent('task_start', { taskId, description: taskDescription })
  }


  function completeTask(taskId) {
    const task = experimentData.tasks.find(t => t.id === taskId)
    if (task) {
      task.endTime = Date.now()
      task.duration = task.endTime - task.startTime
      task.completed = true
    }
    
    
    addInteractionEvent('task_complete', { taskId })
    
    experimentData.currentTaskId = null
  }


  function recordGazeData(data) {
    experimentData.gazeData.push({
      timestamp: Date.now(),
      taskId: experimentData.currentTaskId,
      x: data.x,
      y: data.y,
      relativeTime: Date.now() - experimentData.startTime
    })
  }


  function recordEmotionData(expressions) {
    experimentData.emotionData.push({
      timestamp: Date.now(),
      taskId: experimentData.currentTaskId,
      expressions: expressions,
      relativeTime: Date.now() - experimentData.startTime,
      dominantEmotion: getDominantEmotion(expressions)
    })
  }

  function getDominantEmotion(expressions) {
    return Object.entries(expressions).reduce((a, b) => a[1] > b[1] ? a : b)[0]
  }


  function recordError(error, context = {}) {
    experimentData.errors.push({
      timestamp: Date.now(),
      taskId: experimentData.currentTaskId,
      message: error.message,
      stack: error.stack,
      context: context
    })
  }


  tasksBtn.addEventListener('click', () => {
    modal.style.display = 'block'
  })

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none'
  })

  window.addEventListener('click', (event) => {
    if (event.target === modal) {
      modal.style.display = 'none'
    }
  })

  const video = document.getElementById('video')
  const video2 = document.createElement('video')
  video2.id = 'video2'
  document.querySelector('#container2').appendChild(video2)


  video.width = 480
  video.height = 360
  video2.width = 480
  video2.height = 360


  const setWebGazerSizes = () => {
    const webgazerContainer = document.getElementById('webgazerVideoContainer')
    const webgazerVideo = document.getElementById('webgazerVideoFeed')
    const webgazerOverlay = document.getElementById('webgazerFaceOverlay')
    const webgazerFeedbackBox = document.getElementById('webgazerFaceFeedbackBox')

    if (webgazerContainer) {
      webgazerContainer.style.width = '480px'
      webgazerContainer.style.height = '360px'
    }
    if (webgazerVideo) {
      webgazerVideo.style.width = '480px'
      webgazerVideo.style.height = '360px'
    }
    if (webgazerOverlay) {
      webgazerOverlay.style.width = '480px'
      webgazerOverlay.style.height = '360px'
    }
    if (webgazerFeedbackBox) {
      webgazerFeedbackBox.style.top = '54.4px'
      webgazerFeedbackBox.style.left = '107.7px'
      webgazerFeedbackBox.style.width = '211.2px'
      webgazerFeedbackBox.style.height = '211.2px'
    }
  }


  document.addEventListener('DOMContentLoaded', async () => {
    try {

    await webgazer.setRegression('ridge')
      .setGazeListener((data, timestamp) => {
        if (data == null) {
          console.log('No gaze data');
          return;
        }
        
        const gazeValues = document.getElementById('gazeValues');
        if (gazeValues) {
          gazeValues.textContent = `X: ${Math.round(data.x)}, Y: ${Math.round(data.y)}`;
        }
        
        if (isExperimentRunning) {
          recordGazeData(data);
        }
      })
        .saveDataAcrossSessions(true)
      .begin();


      webgazer.showVideoPreview(true)
        .showPredictionPoints(true)
        .applyKalmanFilter(true);

      console.log('Webgazer initialized successfully');
      

      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('/models')
      ]);
      
      console.log('Face API models loaded successfully');
      startVideo();
      
    } catch (error) {
      console.error('Error during initialization:', error);
      alert('Произошла ошибка при инициализации камеры. Пожалуйста, убедитесь, что у браузера есть доступ к камере.');
    }
    
    const moveWebGazerContainer = () => {
      const webgazerContainer = document.getElementById('webgazerVideoContainer');
      console.log('test');
      if (webgazerContainer) {
        document.querySelector('#container2').appendChild(webgazerContainer)
        setWebGazerSizes()
        
        const loader = document.getElementById('webgazerLoading');
        if (loader) loader.style.display = 'none';
      } else {
        setTimeout(moveWebGazerContainer, 100)
      }
    }

    
    moveWebGazerContainer()

    
    setInterval(setWebGazerSizes, 1000)

    
    function waitForGazeDotAndMotion() {
      let lastPos = null;
      let moved = false;
      let transformLogged = false;
      const loader = document.getElementById('webgazerLoading');
      const check = () => {
        const dot = document.getElementById('webgazerGazeDot');
        if (dot) {
          
          if (!transformLogged && dot.style.transform && dot.style.transform.includes('translate3d')) {
            console.log('webgazerGazeDot получил transform:', dot.style.transform);
            const webgazerVideoContainer = document.getElementById('webgazerVideoContainer');
            if (webgazerVideoContainer) {
              webgazerVideoContainer.style.setProperty('display', 'block', 'important');
            }
            transformLogged = true;
          }
          const pos = { left: dot.style.left, top: dot.style.top };
          if (lastPos && (pos.left !== lastPos.left || pos.top !== lastPos.top)) {
            moved = true;
          }
          lastPos = pos;
          if (moved) {
            if (loader) loader.style.display = 'none';
            return;
          }
        }
        setTimeout(check, 200);
      };
      check();
    }
    waitForGazeDotAndMotion();
  });

  async function checkCameraAccess() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch (error) {
      console.error('Camera access error:', error);
      return false;
    }
  }

  function startVideo() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      checkCameraAccess().then(hasAccess => {
        if (!hasAccess) {
          alert('Нет доступа к камере. Пожалуйста, разрешите доступ к камере в настройках браузера.');
          return;
        }
        
      navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } })
        .then(stream => {
          video.srcObject = stream;
          video2.srcObject = stream;
          
          video.style.transform = 'none';
          
          const webgazerVideo = document.getElementById('webgazerVideoFeed');
          if (webgazerVideo) {
            webgazerVideo.style.transform = 'scaleX(-1)';
          }
            
            console.log('камена успешно включена');
        })
        .catch(err => {
          console.error('Ошибка доступа к камере:', err);
            alert('Произошла ошибка при получении доступа к камере: ' + err.message);
          });
        });
    } else {
      alert('Ваш браузер не поддерживает getUserMedia');
    }
  }

  video.addEventListener('play', () => {
    const canvas = faceapi.createCanvasFromMedia(video)
    canvas.width = 480
    canvas.height = 360
    document.querySelector('#container1').appendChild(canvas)
    const displaySize = { width: 480, height: 360 }
    faceapi.matchDimensions(canvas, displaySize)
    setInterval(async () => {
      const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceExpressions()
      const resizedDetections = faceapi.resizeResults(detections, displaySize)
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
      faceapi.draw.drawDetections(canvas, resizedDetections)
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

      
      if (detections.length > 0) {
        const expressions = detections[0].expressions
        const maxExpression = Object.entries(expressions).reduce((a, b) => a[1] > b[1] ? a : b)
        const faceValues = document.getElementById('faceValues')
        faceValues.textContent = `${maxExpression[0]}: ${(maxExpression[1] * 100).toFixed(1)}%`

        
        if (isExperimentRunning) {
          recordEmotionData(expressions)
        }
      }
    }, 100)
  })

  function calculateAverageTaskDuration() {
    const completedTasks = experimentData.tasks.filter(t => t.completed)
    if (completedTasks.length === 0) return 0
    const totalDuration = completedTasks.reduce((sum, task) => sum + task.duration, 0)
    return totalDuration / completedTasks.length
  }

  function calculateEmotionDistribution() {
    const emotions = {}
    experimentData.emotionData.forEach(data => {
      const dominant = data.dominantEmotion
      emotions[dominant] = (emotions[dominant] || 0) + 1
    })
    return emotions
  }


  document.addEventListener('DOMContentLoaded', () => {
    
    const taskRows = document.querySelectorAll('.tasks-table tbody tr')
    taskRows.forEach((row, index) => {
      const taskId = `task_${index + 1}`
      const startBtn = document.createElement('button')
      startBtn.textContent = 'Начать'
      startBtn.className = 'btn btn-small'
      startBtn.onclick = () => {
        const description = row.cells[1].textContent
        startTask(taskId, description)
        startBtn.textContent = 'Завершить'
        startBtn.disabled = false
        startBtn.onclick = () => {
          completeTask(taskId)
          startBtn.textContent = 'Завершено'
          startBtn.style.backgroundColor = '#4CAF50'
          startBtn.disabled = true
        }
      }
      const buttonCell = document.createElement('td')
      buttonCell.appendChild(startBtn)
      row.appendChild(buttonCell)
    })
    
    const headerRow = document.querySelector('.tasks-table thead tr')
    const actionHeader = document.createElement('th')
    actionHeader.textContent = '  Статус'
    headerRow.appendChild(actionHeader)
  })



  document.getElementById('startCalibrationBtn').onclick = () => {

    document.querySelector('.controls').style.display = 'none';
    document.querySelector('.content-offset').style.display = 'none';
    

    const canvas = document.getElementById('plotting_canvas');
    canvas.style.display = 'block';
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.zIndex = '1001';
    

    PopUpInstruction();
  };


  let selectedJsonFiles = [];


  const uploadJsonBtn = document.getElementById('uploadAnalyzeBtn');
  if (uploadJsonBtn) {
    uploadJsonBtn.textContent = 'Загрузить JSON';
    uploadJsonBtn.removeEventListener && uploadJsonBtn.removeEventListener('click', window._oldUploadAnalyzeHandler);
    window._oldUploadAnalyzeHandler = async function() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json,application/json';
      input.multiple = true;
      input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (!files.length) return;
        for (const file of files) {
          const text = await file.text();
          try {
            const json = JSON.parse(text);
            selectedJsonFiles.push(json);
          } catch (err) {
            alert('Ошибка чтения файла: ' + file.name);
          }
        }
        updateSelectedFilesList();
      };
      input.click();
    };
    uploadJsonBtn.addEventListener('click', window._oldUploadAnalyzeHandler);
  }


  const analyzeBtn = document.getElementById('analyzeBtn');
  if (analyzeBtn) {
    analyzeBtn.removeEventListener && analyzeBtn.removeEventListener('click', window._oldAnalyzeHandler);
    window._oldAnalyzeHandler = async function() {
      if (selectedJsonFiles.length > 0) {
        showLoaderModal();
        try {
          const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experimentDataList: selectedJsonFiles })
          });
          const result = await response.json();
          hideLoaderModal();
          if (result.status === 'success') {
            displayResults(result.results);
          } else {
            alert('Ошибка анализа: ' + (result.error || 'Неизвестная ошибка'));
          }
        } catch (err) {
          hideLoaderModal();
          alert('Ошибка анализа: ' + err.message);
        }
      } else {
        await analyzeSessions();
      }
    };
    analyzeBtn.addEventListener('click', window._oldAnalyzeHandler);
  }


  function updateSelectedFilesList() {
    let list = document.getElementById('selectedFilesList');
    if (!list) {
      list = document.createElement('div');
      list.id = 'selectedFilesList';
      list.style.margin = '10px 0';
      analyzeBtn.parentNode.insertBefore(list, analyzeBtn.nextSibling);
    }
    if (selectedJsonFiles.length === 0) {
      list.style.display = 'none';
    } else {
      list.style.display = 'block';
      if (selectedJsonFiles.length > 4) {
        list.style.maxHeight = '220px';
        list.style.overflowY = 'auto';
      } else {
        list.style.maxHeight = '';
        list.style.overflowY = '';
      }
      list.innerHTML = '<b>Выбранные JSON-файлы для анализа:</b><ul style="margin:4px 0 0 16px;">' +
        selectedJsonFiles.map((f, i) => `<li>Сессия: <code>${f.sessionId || f.user_id || 'без имени'}</code> <button onclick="removeSelectedFile(${i})" style="margin-left:8px;">Удалить</button></li>`).join('') + '</ul>';
    }
  }
  window.removeSelectedFile = function(idx) {
    selectedJsonFiles.splice(idx, 1);
    updateSelectedFilesList();
  };


  function drawGazePoint(x, y) {
    const canvas = document.getElementById('plotting_canvas');
    const ctx = canvas.getContext('2d');
    

    if (!x || !y || isNaN(x) || isNaN(y)) {
      console.log('Invalid coordinates:', x, y);
      return;
    }
    
    console.log('Drawing point at:', x, y); // отладочная информация
    

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    

    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 0, 255, 0.7)';
    ctx.fill();
    

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  }


  window.addEventListener('resize', () => {
    if (document.getElementById('plotting_canvas').style.display === 'block') {
      showCalibrationUI();
    }
  });


  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      generateSessionId,
      formatTime,
      getBrowserName,
      getBrowserVersion,
      clearCalibration,
      calPointClick,
      recordGazeData,
      recordEmotionData,
      startTask,
      completeTask,
      calculatePrecisionPercentages,
      experimentData
    };
  }

  document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('participantModal');
    const form = document.getElementById('participantForm');
    if (modal && form) {
      document.body.style.overflow = 'hidden';
      form.addEventListener('submit', function(e) {
        e.preventDefault();
        const participantNumber = document.getElementById('participantNumber').value.trim();
        const glasses = document.getElementById('glassesCheck').checked;
        const consentData = document.getElementById('consentDataCheck').checked;
        const consentExperiment = document.getElementById('consentExperimentCheck').checked;
        if (!participantNumber || !consentData || !consentExperiment) {
          alert('Пожалуйста, заполните все обязательные поля и дайте согласия.');
          return;
        }

        experimentData.participantNumber = participantNumber;
        experimentData.glasses = glasses;
        experimentData.consentData = consentData;
        experimentData.consentExperiment = consentExperiment;

        window.participantInfo = { participantNumber, glasses, consentData, consentExperiment };

        modal.style.display = 'none';
        document.body.style.overflow = '';
      });
    }
  });


  if (typeof window !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
      const info = localStorage.getItem('participantInfo');
      if (!info) {
        window.location.href = 'participant.html';
        return;
      }
      try {
        const parsed = JSON.parse(info);
        experimentData.participantNumber = parsed.participantNumber;
        experimentData.glasses = parsed.glasses;
        experimentData.consentData = parsed.consentData;
        experimentData.consentExperiment = parsed.consentExperiment;
      } catch (e) {
        window.location.href = 'participant.html';
      }
    });
  }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault();
    }
  });

  const importReportBtn = document.getElementById('importReportBtn');
  if (importReportBtn) {
    importReportBtn.addEventListener('click', function() {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json,application/json';
      input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
          const text = await file.text();
          const json = JSON.parse(text);

          if (json && (json.full_features || json.no_gaze || json.no_emotion)) {
            displayResults(json);
          } else if (json.results && (json.results.full_features || json.results.no_gaze || json.results.no_emotion)) {

            displayResults(json.results);
          } else {
            alert('Файл не похож на отчет анализа!');
          }
        } catch (err) {
          alert('Ошибка чтения файла: ' + err.message);
        }
      };
      input.click();
    });
  }

  function getEmotionColor(emotion) {
    const colors = {
      'happy': '#4CAF50',
      'sad': '#2196F3',
      'angry': '#F44336',
      'surprised': '#FF9800',
      'disgusted': '#9C27B0',
      'fearful': '#795548',
      'neutral': '#607D8B'
    };
    return colors[emotion] || '#9E9E9E';
  }


  document.addEventListener('DOMContentLoaded', () => {

    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.style.display = 'none';
    }

    if (!document.getElementById('individualAnalyzeBtn')) {
        const individualBtn = document.createElement('button');
        individualBtn.id = 'individualAnalyzeBtn';
        individualBtn.textContent = 'Индивидуальный анализ';
        individualBtn.className = 'btn btn-info';
        individualBtn.style.margin = '10px 0 0 10px';
        individualBtn.onclick = () => {
            const sessionIds = prompt('Введите ID двух сессий через запятую (например: session_123, session_456):');
            if (sessionIds) {
                const ids = sessionIds.split(',').map(id => id.trim());
                if (ids.length === 2) {
                    analyzeSessions(ids);
                } else {
                    alert('Пожалуйста, введите ровно два ID сессий через запятую');
                }
            }
        };

        if (analyzeBtn && analyzeBtn.parentNode) {
            analyzeBtn.parentNode.insertBefore(individualBtn, analyzeBtn.nextSibling);
        } else {
            const controls = document.querySelector('.controls');
            if (controls) controls.appendChild(individualBtn);
            else document.body.appendChild(individualBtn);
        }
    }
  });
}