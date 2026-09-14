import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { conversationAPI, aiAPI, userAPI } from '../services/api';
import { getAuthHeaders } from '../services/api';
import RealTimeRecorder from '../components/RealTimeRecorder';
import { AiAvatar } from '../components/AiAvatar';
import { GuajiMascot } from '../components/GuajiMascot';
import { getPersona } from '../config/personaConfig';
import { PracticeReport } from '../components/PracticeReport';
import { AccessibleDialog } from '../components/AccessibleDialog';
import { MessageBubble } from '../components/MessageBubble';
import { useAuth } from '../contexts/AuthContext';
import AudioBar from '../components/AudioBar.jsx';
import {
  collapseAdjacentHistoryDuplicates,
  historyContentKey,
  prepareHistorySnapshot,
  reconcileUserTranscript,
} from '../utils/conversationHistory';
import NetworkAdaptiveManager from '../utils/network-adaptive-manager';
import OptimizedWebSocket from '../utils/websocket-optimized';
import { motion, AnimatePresence } from 'motion/react';
import { useTranslation } from 'react-i18next';
import { resolveDailyLimitModal } from './dailyLimitLogic';
import { shouldUseProgressiveAudio, progressiveAudioSrc, nextProgressiveAttempt } from './audioPlaybackLogic';
import { cleanStreamingText, appendDelta, aiBubbleRenderState, stripAllMarkers, extractMagicSentence } from './streamingTextLogic';
import { normalizeConnectionError, shouldShowConnectionError } from './connectionErrorLogic';
import { calculateTaskProgress, isCompletedWindowEvaluation, isCurrentScoringMessage } from './conversationProgress';
import TaskProgressGuidance from '../components/TaskProgressGuidance';
import { createPcmStreamScheduler, unpackPcmAudioPacket } from '../utils/pcmStreamScheduler';

const MAGIC_TIPS = [
  '点击消息气泡右侧的喇叭图标，可重听 AI 的示范发音。',
  '跟读时保持和 AI 相同的语速和停顿，节奏感是流利度的关键。',
  '背诵时先回想句子结构，再补充词汇细节，效果更好。',
  '遇到长句，可拆成 2-3 个短片段分别练习，再连起来说。',
  '重复 3-5 次才能真正记住一个句型，不用担心次数多。',
  '闭眼想象句子的画面，有助于将语言与情景绑定记忆。',
  '说出来的速度不需要追求完美，意思准确是第一步。',
];

// Default scenario templates — hoisted to module scope so the (large) object
// is allocated once at module load instead of being recreated on every render.
const DEFAULT_SCENARIOS = {
  daily_conversation: [
    { title: "Casual Greetings", tasks: ["Greet someone you just met", "Ask how someone is doing", "Make small talk about the weather"] },
    { title: "Coffee Shop Order", tasks: ["Order your favorite drink", "Ask about menu items", "Request modifications"] },
    { title: "Grocery Shopping", tasks: ["Ask for item locations", "Request quantity and price", "Handle checkout conversation"] },
    { title: "Directions", tasks: ["Ask for directions to a location", "Clarify route details", "Thank for help"] },
    { title: "Phone Call Basics", tasks: ["Answer a phone call properly", "Ask who is calling", "End a call politely"] },
    { title: "Restaurant Dining", tasks: ["Make a reservation", "Order food from menu", "Ask for the bill"] },
    { title: "Public Transport", tasks: ["Ask about schedules", "Buy a ticket", "Confirm your stop"] },
    { title: "Weekend Plans", tasks: ["Discuss weekend activities", "Make suggestions", "Accept or decline invitations"] },
    { title: "Hobbies Discussion", tasks: ["Share your hobbies", "Ask about others' interests", "Make related plans"] },
    { title: "Small Talk (Culture)", tasks: ["Discuss local customs", "Share interesting facts", "Express opinions politely"] }
  ],
  business_meeting: [
    { title: "Self Introduction", tasks: ["Introduce yourself professionally", "Share your role and company", "Exchange contact information"] },
    { title: "Meeting Scheduling", tasks: ["Propose meeting times", "Confirm availability", "Send meeting invites"] },
    { title: "Project Status Update", tasks: ["Summarize current progress", "Discuss blockers", "Plan next steps"] },
    { title: "Client Presentation", tasks: ["Open a presentation", "Explain key points", "Handle Q&A"] },
    { title: "Negotiation Basics", tasks: ["State your position", "Listen to counteroffers", "Reach a compromise"] },
    { title: "Email Discussion", tasks: ["Reference an important email", "Clarify email contents", "Agree on follow-up actions"] },
    { title: "Team Collaboration", tasks: ["Assign tasks to team members", "Check on task progress", "Provide feedback"] },
    { title: "Conference Call", tasks: ["Join a video call", "Share your screen", "Wrap up the call"] },
    { title: "Deadline Management", tasks: ["Discuss timeline constraints", "Request deadline extension", "Commit to new dates"] },
    { title: "Professional Small Talk", tasks: ["Chat about industry news", "Discuss career journeys", "Build rapport"] }
  ],
  travel_survival: [
    { title: "Airport Check-in", tasks: ["Check in for your flight", "Ask about seat preferences", "Handle baggage check"] },
    { title: "Immigration Control", tasks: ["Answer officer questions", "Explain your trip purpose", "Provide required documents"] },
    { title: "Hotel Reservation", tasks: ["Book a room", "Ask about amenities", "Request early check-in"] },
    { title: "Taxi & Rideshare", tasks: ["Request a ride", "Give your destination", "Handle payment"] },
    { title: "Asking Directions", tasks: ["Ask how to get somewhere", "Understand landmark references", "Confirm the route"] },
    { title: "Restaurant Ordering", tasks: ["Ask for recommendations", "Order local cuisine", "Handle dietary requirements"] },
    { title: "Shopping Abroad", tasks: ["Ask prices", "Negotiate or bargain", "Request tax refund info"] },
    { title: "Emergency Situations", tasks: ["Ask for help", "Explain your situation", "Contact emergency services"] },
    { title: "Sightseeing Tours", tasks: ["Book a tour", "Ask tour guide questions", "Express interest or concerns"] },
    { title: "Cultural Small Talk", tasks: ["Discuss local culture", "Share your impressions", "Learn local expressions"] }
  ],
  exam_prep: [
    { title: "Self Introduction (Exam)", tasks: ["Introduce yourself clearly", "Mention your background", "State your goals"] },
    { title: "Describing Pictures", tasks: ["Describe a photo in detail", "Compare two images", "Express your opinion"] },
    { title: "Opinion Questions", tasks: ["State your opinion clearly", "Give supporting reasons", "Conclude your answer"] },
    { title: "Problem Solving", tasks: ["Identify the problem", "Suggest solutions", "Evaluate options"] },
    { title: "Role-play Scenarios", tasks: ["Understand the situation", "Respond appropriately", "Handle follow-ups"] },
    { title: "Discussion & Debate", tasks: ["Express agreement/disagreement", "Build on others' points", "Summarize the discussion"] },
    { title: "Long Turn Speaking", tasks: ["Speak for 1-2 minutes fluently", "Structure your answer", "Manage your time"] },
    { title: "Pronunciation Practice", tasks: ["Practice difficult sounds", "Work on intonation", "Reduce accent interference"] },
    { title: "Vocabulary Expansion", tasks: ["Use academic vocabulary", "Explain complex terms", "Paraphrase effectively"] },
    { title: "Mock Exam Practice", tasks: ["Complete a timed practice", "Self-evaluate performance", "Identify improvement areas"] }
  ],
  presentation: [
    { title: "Opening Strong", tasks: ["Grab audience attention", "Introduce your topic", "Preview main points"] },
    { title: "Explaining Data", tasks: ["Present statistics clearly", "Interpret chart information", "Draw conclusions"] },
    { title: "Storytelling", tasks: ["Share a relevant story", "Connect to your message", "Engage emotionally"] },
    { title: "Handling Q&A", tasks: ["Listen carefully to questions", "Provide clear answers", "Handle difficult questions"] },
    { title: "Visual Aid Description", tasks: ["Reference your slides", "Explain diagrams", "Guide audience attention"] },
    { title: "Transitions", tasks: ["Move between topics smoothly", "Recap previous points", "Preview next sections"] },
    { title: "Persuasion Techniques", tasks: ["Present your argument", "Address counter-arguments", "Call to action"] },
    { title: "Closing Impact", tasks: ["Summarize key takeaways", "End with a memorable statement", "Thank your audience"] },
    { title: "Team Presentation", tasks: ["Coordinate with co-presenters", "Handle handoffs", "Support each other"] },
    { title: "Impromptu Speaking", tasks: ["Speak on unexpected topics", "Organize thoughts quickly", "Deliver confidently"] }
  ]
};

function stripAIMarkers(text) {
  if (!text) return text;
  return stripAllMarkers(text).trim();
}

// Determine whether to suppress auto-play for a given AI message at index `idx`.
// Default: COS URL triggers auto-play (audioPlayed=false) to guarantee playback
// even when streaming PCM chunks failed to play (autoplay policy, decode error,
// network drop, etc). The auto-play handler stops in-flight streaming audio
// before playing the COS URL, preventing double playback.
// Only suppress for the muted welcome message after a refresh/retry.
// Hoisted to module scope (takes welcomeMuted + messages as explicit args) so it
// is allocated once instead of being rebuilt on every message append/setMessages.
function shouldSuppressAutoPlay(welcomeMuted, messages, idx) {
  if (welcomeMuted) {
    const isFirst = idx === 0 || (idx === 1 && messages[0]?.type === 'system');
    if (isFirst) return true;
  }
  return false;
}

// Split text into sentence-sized chunks for the rolling CC caption.
// Handles CJK punctuation (。！？) and Latin (.!?). Preserves the closing
// punctuation. No length-based merging — AI may produce any number of
// sentences per turn (2, 3, 4, dozens) and every sentence should appear on
// screen. Only drops fragments that have no readable content.
function splitIntoSentences(text) {
  if (!text) return [];
  const raw = text.match(/[^。！？!?.]+[。！？!?.]?/g) || [text];
  return raw
    .map(s => s.trim())
    .filter(s => /[\p{L}\p{N}]/u.test(s)); // keep only fragments containing a letter or digit
}

// Rolling caption that advances through `text`'s sentences in sync with TTS
// playback progress. `getProgressRatio()` is read from a rAF loop so the UI
// stays smooth without re-rendering on every audio chunk.
function CCRollingCaption({ isAISpeaking, text, getProgressRatio }) {
  const [sentenceIdx, setSentenceIdx] = React.useState(0);
  const sentences = React.useMemo(() => splitIntoSentences(text), [text]);

  React.useEffect(() => {
    if (!isAISpeaking || sentences.length <= 1) {
      setSentenceIdx(0);
      return;
    }
    let raf;
    const tick = () => {
      const ratio = getProgressRatio();
      const idx = Math.min(sentences.length - 1, Math.floor(ratio * sentences.length));
      // Monotonic: never roll backwards. New TTS chunks arriving mid-turn
      // grow `total`, which can briefly shrink ratio — clamp so the caption
      // only advances. Resets to 0 on next turn via the early-return above.
      setSentenceIdx(prev => (idx > prev ? idx : prev));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isAISpeaking, sentences, getProgressRatio]);

  if (!isAISpeaking || sentences.length === 0) return null;
  const current = sentences[Math.min(sentenceIdx, sentences.length - 1)];
  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      style={{
      marginTop: 16, padding: '10px 18px', borderRadius: 14,
      background: 'rgba(0,0,0,0.72)', color: '#fff',
      fontSize: 14, lineHeight: 1.5, maxWidth: '85%', textAlign: 'center', fontWeight: 500,
      animation: 'subtitle-in 240ms ease-out',
      minHeight: '44px', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      key={sentenceIdx}
    >
      <span>{current}</span>
    </div>
  );
}

function DailyQAPassModal({ onClose, onReturn, isBonus }) {
  const { t } = useTranslation();
  return (
    <div style={{
      position: 'fixed', bottom: 80, left: '50%', transform: 'translateX(-50%)',
      background: '#FFFFFF', borderRadius: 20, padding: '20px 28px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.18)', zIndex: 300,
      maxWidth: 360, width: '90%', textAlign: 'center',
      animation: 'slideUpBanner 0.4s ease-out'
    }}>
      <div style={{ fontSize: 48, marginBottom: 8 }}>{isBonus ? '👏' : '✅'}</div>
      <h3 style={{ fontSize: 18, fontWeight: 700, color: '#1F2937', marginBottom: 6 }}>
        {isBonus ? t('daily_qa_pass_bonus_title') : t('daily_qa_pass_title')}
      </h3>
      <p style={{ fontSize: 13, color: '#6B7280', marginBottom: 16 }}>
        {isBonus ? t('daily_qa_pass_bonus_desc') : t('daily_qa_pass_desc')}
      </p>
      <button
        onClick={onReturn}
        style={{
          width: '100%', padding: '12px 0', borderRadius: 12,
          background: 'linear-gradient(135deg, #637FF1, #a47af6)', color: '#fff',
          fontWeight: 700, fontSize: 15, border: 'none', cursor: 'pointer', marginBottom: 8
        }}>
        {t('daily_qa_pass_return')}
      </button>
      <button
        onClick={onClose}
        style={{
          background: 'transparent', border: 'none', color: '#9CA3AF',
          fontSize: 13, cursor: 'pointer', padding: '4px 0'
        }}>
        {t('daily_qa_pass_continue')}
      </button>
      <style>{`
        @keyframes slideUpBanner {
          0% { transform: translateX(-50%) translateY(30px); opacity: 0; }
          100% { transform: translateX(-50%) translateY(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

function ScorePopup({ scores, delta, onClose }) {
  const scoreValue = (...keys) => {
    const value = keys.map(key => scores?.[key]).find(candidate => candidate != null);
    const numeric = Number(value);
    return Number.isFinite(numeric) ? Math.max(0, Math.min(10, numeric)) : 5;
  };
  const fluency = scoreValue('fluency');
  const grammar = scoreValue('grammar', 'grammar_quality');
  const vocabulary = scoreValue('vocabulary', 'keyword_coverage');
  const relevance = scoreValue('task_relevance', 'topic_relevance');
  const overall = Math.round(
    (fluency + grammar + vocabulary + relevance) / 4 * 10
  );
  const circumference = 2 * Math.PI * 45;
  const offset = circumference * (1 - Math.min(overall, 100) / 100);
  const dims = [
    { label: '流利度', val: Math.round(fluency * 10) },
    { label: '语法',   val: Math.round(grammar * 10) },
    { label: '词汇',   val: Math.round(vocabulary * 10) },
    { label: '话题',   val: Math.round(relevance * 10) },
  ];
  return (
    <AccessibleDialog
      title={`本轮评估，增加 ${delta} 熟练度`}
      description="查看本轮口语练习的四项评分"
      onClose={onClose}
      closeLabel="关闭本轮评估"
      showCloseButton={false}
      panelClassName="app-modal-boundary !w-[min(90vw,340px)] !rounded-[29px] !bg-slate-800 !text-white"
      zIndex={260}
    >
      <div style={{ padding:32, textAlign:'center' }}>
        <h2 style={{ color:'#F8FAFC', marginBottom:24, fontSize:18, fontWeight:700 }}>本轮评估 +{delta} 熟练度 🎉</h2>
        <div style={{ width:120, height:120, margin:'0 auto 24px', position:'relative' }}>
          <svg viewBox="0 0 100 100" style={{ transform:'rotate(-90deg)', width:'100%', height:'100%' }}>
            <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="8"/>
            <circle cx="50" cy="50" r="45" fill="none" stroke="#10B981" strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    style={{ transition:'stroke-dashoffset 1s ease' }}/>
          </svg>
          <span style={{ position:'absolute', top:'50%', left:'50%',
                         transform:'translate(-50%,-50%)',
                         fontSize:32, fontWeight:700, color:'#10B981' }}>{overall}</span>
        </div>
        <div style={{ textAlign:'left', marginBottom:24 }}>
          {dims.map(({ label, val }) => (
            <div key={label} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
              <span style={{ width:40, fontSize:12, color:'#94A3B8', flexShrink:0 }}>{label}</span>
              <div
                role="progressbar"
                aria-label={`${label}评分`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={val}
                style={{ flex:1, height:6, background:'#334155', borderRadius:3, overflow:'hidden' }}
              >
                <div style={{ height:'100%', width:`${val}%`, background:'#637FF1',
                              borderRadius:3, transition:'width 1s ease' }}/>
              </div>
              <span style={{ width:28, fontSize:12, color:'#F8FAFC', textAlign:'right', flexShrink:0 }}>{val}</span>
            </div>
          ))}
        </div>
        <button onClick={onClose}
                className="min-h-11 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
                style={{ width:'100%', padding:'12px', borderRadius:20,
                         background:'#637FF1', color:'#fff', border:'none',
                         fontWeight:600, cursor:'pointer', fontSize:14 }}>
          继续练习
        </button>
      </div>
    </AccessibleDialog>
  );
}

function TaskCompletionSheet({ taskReadyToComplete, tasks, completedTasks, onConfirm, onContinue, canConfirm = true }) {
  const completedTitle = taskReadyToComplete?.task_title || '';
  const taskList = tasks || [];
  const totalCount = taskList.length || (completedTasks?.size || 0) + 1;
  const completedCount = Math.min(totalCount, (completedTasks?.size || 0) + 1);

  const completedTaskIndex = taskList.findIndex(task => {
    if (typeof task === 'object' && taskReadyToComplete?.task_id != null) {
      return String(task.id) === String(taskReadyToComplete.task_id);
    }
    const text = typeof task === 'string' ? task : task.text;
    return text?.trim() === completedTitle.trim();
  });
  const remainingTasks = completedTaskIndex >= 0
    ? taskList.slice(completedTaskIndex + 1)
    : taskList;
  const nextTask = remainingTasks.find(task => {
    const text = typeof task === 'string' ? task : task.text;
    return text?.trim() !== completedTitle.trim() && !completedTasks?.has(text);
  });
  const nextTitle = typeof nextTask === 'string' ? nextTask : (nextTask?.text || '');

  return (
    <AccessibleDialog
      title="当前子任务已达标"
      description={nextTitle ? `即将切换到下一个任务：${nextTitle}` : '可以完成当前任务或继续深入练习'}
      onClose={onContinue}
      closeLabel="关闭任务完成提示"
      placement="bottom"
      showCloseButton={false}
      overlayClassName="!p-0 sm:!p-4"
      panelClassName="!max-w-[440px] !rounded-t-3xl sm:!rounded-3xl !bg-transparent"
      zIndex={250}
    >
      <motion.div
        initial={{ y: '100%' }}
        animate={{ y: 0 }}
        exit={{ y: '100%' }}
        transition={{ type: 'spring', damping: 28, stiffness: 320 }}
        className="rounded-t-3xl sm:rounded-3xl"
        style={{
          width: '100%', maxWidth: 440,
          background: 'linear-gradient(135deg, #637FF1 0%, #8B5CF6 100%)',
          padding: '28px 24px max(32px, env(safe-area-inset-bottom))',
          color: '#fff',
        }}
      >
        {/* Drag handle */}
        <div style={{ width: 40, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.3)', margin: '0 auto 20px' }} />

        {/* Completed task */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          background: 'rgba(255,255,255,0.15)', borderRadius: 16, padding: '14px 16px', marginBottom: 12,
        }}>
          <div style={{
            width: 40, height: 40, borderRadius: 12, background: 'rgba(255,255,255,0.25)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20, flexShrink: 0,
          }}>✅</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 2 }}>任务完成</div>
            <div style={{ fontSize: 15, fontWeight: 600, lineHeight: 1.45, overflowWrap: 'anywhere' }}>
              {completedTitle}
            </div>
          </div>
          <div style={{ fontSize: 13, opacity: 0.8, flexShrink: 0 }}>{completedCount}/{totalCount}</div>
        </div>

        {/* Next task preview */}
        {nextTitle && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            background: 'rgba(255,255,255,0.1)', borderRadius: 16, padding: '14px 16px', marginBottom: 24,
          }}>
            <div style={{
              width: 40, height: 40, borderRadius: 12, background: 'rgba(255,255,255,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20, flexShrink: 0,
            }}>🎯</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 2 }}>下一个任务</div>
              <div style={{ fontSize: 15, fontWeight: 600, lineHeight: 1.45, overflowWrap: 'anywhere' }}>
                {nextTitle}
              </div>
            </div>
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <button
            onClick={onConfirm}
            disabled={!canConfirm}
            aria-describedby={!canConfirm ? 'task-completion-connection-note' : undefined}
            className="min-h-11 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80 disabled:cursor-not-allowed disabled:opacity-60"
            style={{
              width: '100%', padding: '14px', borderRadius: 16,
              background: '#fff', color: '#637FF1', border: 'none',
              fontWeight: 700, fontSize: 15, cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            }}
          >
            <span>🚀</span>
            <span>{canConfirm ? (nextTitle ? '切换下一个任务' : '完成当前任务') : '连接已断开，请先重连'}</span>
          </button>
          {!canConfirm && (
            <p id="task-completion-connection-note" role="status" style={{ margin: '-2px 0 2px', textAlign: 'center', fontSize: 12, color: 'rgba(255,255,255,0.9)' }}>
              当前进度仍会保留，连接恢复后即可切换。
            </p>
          )}
          <button
            onClick={onContinue}
            className="min-h-11 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
            style={{
              width: '100%', padding: '14px', borderRadius: 16,
              background: 'rgba(255,255,255,0.15)', color: '#fff', border: '1px solid rgba(255,255,255,0.25)',
              fontWeight: 600, fontSize: 15, cursor: 'pointer',
            }}
          >
            继续深入当前任务
          </button>
        </div>
      </motion.div>
    </AccessibleDialog>
  );
}

function Conversation() {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();
  const { user, token, loading } = useAuth(); // Added loading state
  const scenarioEmoji = location.state?.emoji;
  const persona = getPersona(localStorage.getItem('ai_voice') || 'Tina');

  // UI States
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isRestoringSession, setIsRestoringSession] = useState(false);
  const [currentRole, setCurrentRole] = useState('OralTutor'); // Default role
  const [isAISpeaking, setIsAISpeaking] = useState(false);
  const [isUserRecording, setIsUserRecording] = useState(false);
  // Flips true when user releases the recorder; flips false the moment the
  // first audio frame from this AI turn starts playing. Drives the mascot
  // `thinking` expression during the request→response gap (test case 6.x).
  const [isWaitingForAIResponse, setIsWaitingForAIResponse] = useState(false);
  const [webSocketError, setWebSocketError] = useState(null);
  // Set true when the backend explicitly rejects this connection (e.g.
  // "Invalid scenario" error frame, or a 1008/4400 close). Guards the close
  // handler from clearing/overwriting the user-facing error and from
  // auto-reconnecting into the same rejection loop.
  const wsRejectedRef = useRef(false);
  // Mirror of wsRejectedRef for render: when the connection is rejected we show
  // a "back to Discover" exit instead of a (futile) retry button.
  const [wsRejected, setWsRejected] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const sessionIdRef = useRef(null);
  const [selection, setSelection] = useState({ text: '', x: 0, y: 0, visible: false });
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [playingAudioUrl, setPlayingAudioUrl] = useState(null);
  const [taskBarFaded, setTaskBarFaded] = useState(false);
  const taskBarFadeTimerRef = useRef(null);
  const [welcomeMessageShown, setWelcomeMessageShown] = useState(false); // Track if welcome message has been shown
  const connectWebSocketRef = useRef(null); // Ref to store connectWebSocket function
  
  // WebSocket connection control states
  const [isManualDisconnect, setIsManualDisconnect] = useState(false); // Track if user manually disconnected
  const [reconnectAttempts, setReconnectAttempts] = useState(0); // Track reconnection attempts
  const MAX_RECONNECT_ATTEMPTS = 5; // Backoff schedule: 1, 2, 4, 8, 10 seconds
  // Ref mirrors of the two reconnect-control states. The close-handler's
  // setTimeout (and connectWebSocket's useCallback, which omits these from its
  // deps) would otherwise read stale snapshots — causing a spurious reconnect
  // when the user manually retries during the backoff window. Read .current.
  const isManualDisconnectRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef(null); // pending auto-reconnect setTimeout id; cleared on manual retry / reject
  const isRestoringSessionRef = useRef(false);

  // Default scenario templates are hoisted to module scope (see DEFAULT_SCENARIOS above).

  // localStorage key helper — encodes scenario name to prevent key injection via crafted URLs
  const _lsScenarioKey = (prefix, raw) => `${prefix}${encodeURIComponent(raw || '')}`;
  const _lsSessionKey = (userId, scenario) => `session_${encodeURIComponent(String(userId || 'anon'))}_${encodeURIComponent(scenario || '')}`;

  // Scenario Tasks State
  // Initialize as empty - will be populated from backend in useEffect
  const [tasks, setTasks] = useState([]);
  const [completedTasks, setCompletedTasks] = useState(new Set());
  
  // Task Progress State (for progress bar)
  const [currentTaskProgress, setCurrentTaskProgress] = useState(() => {
    // Try to load from localStorage for persistence
    const searchParams = new URLSearchParams(window.location.search);
    const scenario = searchParams.get('scenario') || location.state?.scenario;
    if (scenario) {
      const saved = localStorage.getItem(_lsScenarioKey('task_progress_', scenario));
      return saved ? parseInt(saved, 10) : 0;
    }
    return 0;
  });
  const [currentTaskScore, setCurrentTaskScore] = useState(0);
  const [engagementLevel, setEngagementLevel] = useState('中');
  const [showScorePopup, setShowScorePopup] = useState(false);
  const [batchScores, setBatchScores] = useState(null);
  const [latestDelta, setLatestDelta] = useState(0); // 高/中/低
  const previousProgressRef = useRef(0); // Track previous progress to prevent unreasonable jumps
  const lastSeenTaskIdRef = useRef(null); // Track task ID to detect task switches
  const scoringGenerationByTaskRef = useRef(new Map()); // Reject late evaluations from before an explicit reset
  const [progressFeedback, setProgressFeedback] = useState(null);
  const [completionSheetDismissed, setCompletionSheetDismissed] = useState(false);
  const feedbackOrderRef = useRef(new Map());
  const activeScoringTask = tasks.find(task => typeof task === 'object'
    && task.status !== 'completed' && !completedTasks.has(task.text));
  const activeScoringTaskRef = useRef(null);
  activeScoringTaskRef.current = activeScoringTask;
  const acceptScoringMessage = useCallback(payload => isCurrentScoringMessage(
    payload, activeScoringTaskRef.current, scoringGenerationByTaskRef.current,
    feedbackOrderRef.current.get(`${payload?.task_id}:${payload?.scoring_generation ?? 0}`) || 0,
  ), []);

  // Initialize showTasks based on whether we have scenario info
  // Tasks will be loaded from backend, so we show tasks if scenario is specified
  const [showTasks, setShowTasks] = useState(() => {
    const searchParams = new URLSearchParams(window.location.search);
    const scenarioFromUrl = searchParams.get('scenario');
    const scenarioFromState = location.state?.scenario;
    // Show tasks if we have scenario info (tasks will be loaded from backend)
    return !!scenarioFromUrl || !!scenarioFromState;
  });
  
  // Track if tasks are loading to prevent showing "Loading tasks" when we know tasks exist
  const [tasksLoading, setTasksLoading] = useState(false);
  
  // CC (immersive) mode — shows GuajiMascot overlay
  const [ccMode, setCcMode] = useState(false);

  useEffect(() => {
    if (!ccMode) return undefined;
    const exitOnEscape = (event) => {
      if (event.key === 'Escape') setCcMode(false);
    };
    document.addEventListener('keydown', exitOnEscape);
    return () => document.removeEventListener('keydown', exitOnEscape);
  }, [ccMode]);

  // Scenario Completion State
  const [showCompletionModal, setShowCompletionModal] = useState(false);
  const [scenarioScore, setScenarioScore] = useState(0);
  const [scenarioReviewData, setScenarioReviewData] = useState(null); // Store review data for AI feedback
  const [allScenarios, setAllScenarios] = useState([]);
  const [currentScenarioIndex, setCurrentScenarioIndex] = useState(0);
  const [currentScenarioTitle, setCurrentScenarioTitle] = useState('');
  const completionCheckedRef = useRef(false); // Prevent duplicate modal triggers
  const hasViewedCompletionModalRef = useRef(false); // Track if user has already viewed and closed the modal
  const pendingCompletionModalRef = useRef(false); // Modal wants to open but is waiting on scenario_review WS
  const completionFallbackTimerRef = useRef(null); // Hard-timeout that opens the modal even if review never lands

  // 双阶段 UI State（Magic Repetition 和 Scene Theater）
  // useRef 保证只在首次挂载时读取 URL，避免每次 render 重新解析
  const isRecallMode = useRef(new URLSearchParams(window.location.search).get('mode') === 'recall').current;
  const isDailyQAMode = useRef(new URLSearchParams(window.location.search).get('mode') === 'daily_qa').current;
  // Onboarding Tour demo: highlight the mic UI only — no WS, no AI calls.
  const isTourMode = useRef(new URLSearchParams(window.location.search).get('mode') === 'tour').current;
  const [currentPhase, setCurrentPhase] = useState(isRecallMode ? 'magic_repetition' : 'scene_theater');
  const currentPhaseRef = useRef(isRecallMode ? 'magic_repetition' : 'scene_theater');
  const [sceneImageUrl, setSceneImageUrl] = useState(null);
  const [magicPassedTasks, setMagicPassedTasks] = useState(() => {
    try {
      const sc = new URLSearchParams(window.location.search).get('scenario') || '';
      const stored = sc && localStorage.getItem(_lsScenarioKey('magic_passed_', sc));
      return stored ? new Set(JSON.parse(stored)) : new Set();
    } catch { return new Set(); }
  }); // task indices that passed magic（localStorage 持久化，key: magic_passed_{scenario}）
  const [theaterCompletedTasks, setTheaterCompletedTasks] = useState(new Set());
  const [aiFeedback, setAiFeedback] = useState('');
  const [dailyScenariosUsed, setDailyScenariosUsed] = useState(0);
  const [currentMagicSentence, setCurrentMagicSentence] = useState(() => {
    try {
      const sc = new URLSearchParams(window.location.search).get('scenario') || '';
      return sc ? (localStorage.getItem(_lsScenarioKey('magic_sentence_', sc)) || '') : '';
    } catch { return ''; }
  }); // 魔法重复阶段当前需复述的句子（持久化到 localStorage，key: magic_sentence_{scenario}）
  const [magicCardState, setMagicCardState] = useState('waiting'); // 'waiting'|'reciting'|'passed'
  const [magicCardCovered, setMagicCardCovered] = useState(false);
  const [isPeeking, setIsPeeking] = useState(false);
  const [showSkipButton, setShowSkipButton] = useState(false);
  const [tipIndex, setTipIndex] = useState(0);

  // Daily QA Mode State
  const [dailyQAQuestion, setDailyQAQuestion] = useState(null);
  const [dailyQAError, setDailyQAError] = useState(false);
  const [showDailyQAPassModal, setShowDailyQAPassModal] = useState(false);
  // Language gate warning surfaces a visible toast when the backend detects
  // the user answered in the wrong script (e.g. Chinese reply to an English
  // daily QA). Auto-dismisses after 6s.
  const [languageGateWarning, setLanguageGateWarning] = useState(null);
  const [dailyQAIsBonus, setDailyQAIsBonus] = useState(false);
  const [dailyQAReferenceAnswer, setDailyQAReferenceAnswer] = useState('');
  const [showReferenceAnswer, setShowReferenceAnswer] = useState(false);
  const navTimeoutRef = useRef(null);

  // Task Completion Confirmation State
  const [taskReadyToComplete, setTaskReadyToComplete] = useState(null); // shape: { task_id, task_title } | null
  const [taskCompletionPending, setTaskCompletionPending] = useState(false);

  // 每日对话轮次上限 — daily_limit_reached 事件弹出的模态
  const [dailyLimitModal, setDailyLimitModal] = useState(null);

  const getScoreFeedback = (score, reviewData = null) => {
    const stripEmoji = (s) => s.replace(
      /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, ''
    ).trim();

    if (reviewData) {
        const parts = [];

        // 1. analysis.summary — 最个性化：含实际轮数、任务数、平均分，后端已语言感知
        const summary = reviewData.analysis?.summary;
        if (summary && typeof summary === 'string') {
            parts.push(stripEmoji(summary));
        }

        // 2. 首条 recommendations — 具体改进建议，后端已按 native_language 生成
        const recs = reviewData.recommendations;
        if (Array.isArray(recs) && recs.length > 0) {
            const rec = stripEmoji(recs[0]);
            // 只在与 summary 不同时追加，避免重复
            if (rec && rec !== parts[0]) {
                parts.push(rec);
            }
        }

        if (parts.length > 0) {
            return { emoji: '', text: parts.join('\n'), level: 'excellent' };
        }
    }

    // Fallback：无 reviewData 时按分数给出简洁反馈
    if (score >= 90) return { emoji: '', text: '表现优秀，表达流利自然，词汇使用准确。', level: 'excellent' };
    if (score >= 75) return { emoji: '', text: '表现良好，表达清晰准确，可继续练习复杂句型。', level: 'good' };
    if (score >= 60) return { emoji: '', text: '进步明显，建议多练习口语表达的流畅度。', level: 'fair' };
    return { emoji: '', text: '建议继续练习，多听多说以提高表达能力。', level: 'needsWork' };
  };

  // Report practice time on unmount
  useEffect(() => {
    return () => {
      if (practiceStartTimeRef.current) {
        const minutes = Math.round((Date.now() - practiceStartTimeRef.current) / 60000);
        practiceStartTimeRef.current = null;
        if (minutes > 0 && minutes <= 120) {
          userAPI.recordPracticeTime(minutes).catch(() => {});
        }
      }
    };
  }, []);

  // 查询每日场景数（mount 时，使用 localStorage）
  useEffect(() => {
    const today = new Date().toISOString().slice(0, 10); // '2026-03-30'
    const key = `daily_scenarios_${today}`;
    const count = parseInt(localStorage.getItem(key) || '0', 10);
    setDailyScenariosUsed(Math.min(count, 3));
  }, []);

  // Initialize completed tasks set and check for scenario completion
  useEffect(() => {
      console.log('Initializing tasks state:', tasks); // Debug log
      console.log('All scenarios length:', allScenarios.length); // Debug log
      console.log('Tasks array details:', JSON.stringify(tasks)); // Debug log
      
      if (tasks.length > 0) {
          const completed = new Set();
          let totalScore = 0;
          let objectTaskCount = 0;
          let completedCount = 0;

          tasks.forEach(t => {
              if (typeof t === 'object') {
                  objectTaskCount++;
                  if (t.status === 'completed') {
                      completed.add(t.text);
                      totalScore += (t.score || 0);
                      completedCount++;
                  }
              }
          });
          setCompletedTasks(completed);
          
          // Only show tasks if there are tasks and they haven't been completed yet
          const hasIncompleteTasks = objectTaskCount > 0 && completedCount < objectTaskCount;
          setShowTasks(hasIncompleteTasks);
          
          // Calculate average score for completed tasks
          if (completedCount > 0) {
              setScenarioScore(Math.round(totalScore / completedCount));
          }

          // Check if all tasks are completed to show completion modal
          // Only show if user hasn't already viewed and closed the modal
          if (objectTaskCount > 0 && completedCount === objectTaskCount &&
              !completionCheckedRef.current && !hasViewedCompletionModalRef.current) {
              completionCheckedRef.current = true;

              // 每日场景计数 +1 —— 在此处（completionCheckedRef 守卫，每个场景仅触发
              // 一次）记账，而非监听 showCompletionModal 可见性。后者会在弹窗多次
              // 开关 / re-render 时重复 +1，导致「完成 1 场景却记 3 场景」的虚高。
              try {
                  const today = new Date().toISOString().slice(0, 10);
                  const dkey = `daily_scenarios_${today}`;
                  const prev = parseInt(localStorage.getItem(dkey) || '0', 10);
                  const next = Math.min(prev + 1, 3);
                  localStorage.setItem(dkey, String(next));
                  setDailyScenariosUsed(next);
              } catch {}


              // Fetch scenario review for personalized AI feedback, THEN open the
              // completion modal — but only once the AI commentary is actually
              // available. The old code opened the modal on a fixed 1s timer,
              // which fired well before the backend's scenario_review WS event
              // (two serial ~20s LLM calls), so the report showed an empty
              // 「详细反馈」. Now: if the REST review is ready we open immediately;
              // otherwise we mark the open as pending and let the WS
              // scenario_review handler (which sets scenarioReviewData) trigger
              // the open via the effect below. A hard timeout still opens the
              // modal so the user is never stuck waiting if the review never lands.
              const clearMagicProgress = () => {
                  try {
                      const sc = new URLSearchParams(window.location.search).get('scenario') || '';
                      if (sc) localStorage.removeItem(_lsScenarioKey('magic_passed_', sc));
                      if (sc) localStorage.removeItem(_lsScenarioKey('magic_sentence_', sc));
                  } catch {}
              };
              const fetchReviewAndShowModal = async () => {
                  clearMagicProgress();
                  // 立即打开报告 —— 不再等慢 WS（~20s 双 LLM 调用）。PracticeReport 在
                  // reviewData 缺失时用技能分数兜底渲染「详细反馈」，等 REST/WS 的真
                  // 点评到达后 setScenarioReviewData 触发 re-render 自动补全，无空白、无卡顿。
                  setShowCompletionModal(true);
                  try {
                      const review = await userAPI.getScenarioReview(currentScenarioTitle);
                      if (review) {
                          console.log('📊 Fetched scenario review (REST):', review);
                          setScenarioReviewData(review);
                      }
                      // REST 为空时（首次完成，DB 尚未写入），WS scenario_review 事件会
                      // 稍后送达并 setScenarioReviewData —— 报告已打开，届时自动补全点评。
                  } catch (error) {
                      console.error('Failed to fetch scenario review:', error);
                  }
              };
              fetchReviewAndShowModal();
          }
      }
  }, [tasks, allScenarios]);

  // 同步 currentPhase → ref（供 handleJsonMessage 等 callback 读取）
  useEffect(() => { currentPhaseRef.current = currentPhase; }, [currentPhase]);

  // 持久化 currentMagicSentence → localStorage（刷新页面后可恢复，key: magic_sentence_{scenario}）
  useEffect(() => {
    try {
      const sc = new URLSearchParams(window.location.search).get('scenario') || '';
      if (!sc) return;
      if (currentMagicSentence) {
        localStorage.setItem(_lsScenarioKey('magic_sentence_', sc), currentMagicSentence);
      } else {
        localStorage.removeItem(_lsScenarioKey('magic_sentence_', sc));
      }
    } catch {}
  }, [currentMagicSentence]);

  // Tips 轮播（魔法重复 waiting 阶段每 9s 切换）
  useEffect(() => {
    if (currentPhase !== 'magic_repetition' || magicCardState !== 'waiting') return;
    const timer = setInterval(() => {
      setTipIndex(prev => (prev + 1) % MAGIC_TIPS.length);
    }, 9000);
    return () => clearInterval(timer);
  }, [currentPhase, magicCardState]);

  // 当 scenario_review（WS）迟到送达 AI 点评时，若完成弹窗在等待中则立即打开，
  // 并清掉硬超时兜底。这样报告打开时「详细反馈」已有内容，不再空标题。
  useEffect(() => {
    if (scenarioReviewData && pendingCompletionModalRef.current) {
      pendingCompletionModalRef.current = false;
      if (completionFallbackTimerRef.current) {
        clearTimeout(completionFallbackTimerRef.current);
        completionFallbackTimerRef.current = null;
      }
      setShowCompletionModal(true);
    }
  }, [scenarioReviewData]);

  // 卸载时清理兜底定时器
  useEffect(() => () => {
    if (completionFallbackTimerRef.current) clearTimeout(completionFallbackTimerRef.current);
  }, []);

  // 注：每日场景计数已移至场景完成检测处（completionCheckedRef 守卫，每场景仅 +1），
  // 不再监听 showCompletionModal 可见性——避免弹窗重复开关导致计数虚高。

  // Audio context and refs
  const audioContextRef = useRef(null);
  const nextStartTimeRef = useRef(0);
  // Progressive playback of COS replay audio uses a plain HTMLAudioElement
  // (edge-download-and-play, first byte out) instead of Web Audio's
  // fetch-whole-file-then-decode. Kept in a ref so stopAudioPlayback can pause
  // and clear it, and so the CC caption can read its currentTime/duration.
  const htmlAudioRef = useRef(null);
  // CC caption rolling: track when current AI speech started + total duration
  // at that moment, so we can compute "played ratio" and pick which sentence
  // should be on screen. Reset on each new turn (when isAISpeaking flips
  // false→true).
  const speechStartTimeRef = useRef(0);
  const speechTotalDurationRef = useRef(0);
  const audioQueueRef = useRef([]);
  const pcmSchedulerRef = useRef(null);
  const playAudioChunkRef = useRef(null);
  const pendingStreamAudioRef = useRef([]);
  const receivedStreamAudioRef = useRef(false);
  const aiTextReadyForAudioRef = useRef(false);
  const activeAudioResponseIdRef = useRef(null);
  const streamAudioDoneRef = useRef(false);
  // Tracks whether streaming PCM chunks have actually played for the CURRENT turn,
  // measured from the last stopAudioPlayback() cut point (which is the natural
  // per-turn boundary: user recording start / auto-play start / magic_pass).
  // Used to de-dupe the opening greeting (and any turn) when the streaming path
  // already produced audible audio but the slow COS audio_url arrives AFTER the
  // short streaming queue already drained — at which point stopAudioPlayback()
  // stops an empty queue and the COS URL would play a SECOND time.
  // Set true in playAudioChunk; reset to false in stopAudioPlayback.
  const streamedAudioSinceCutRef = useRef(false);
  const isInterruptedRef = useRef(false);
  const currentUserMessageIdRef = useRef(null);
  const currentRecordingSessionIdRef = useRef(null); // Track current recording session to ignore cancelled audio
  const messagesEndRef = useRef(null);
  const socketRef = useRef(null);
  const lastProficiencyUpdateRef = useRef(null); // Track last processed proficiency update to prevent duplicates
  const recorderRef = useRef(null); // Ref for RealTimeRecorder to control session ID
  const practiceStartTimeRef = useRef(null); // Set on first user recording, not on WS open
  const historyAutosaveTimerRef = useRef(null);
  const restoredAiContentKeysRef = useRef(new Set());
  const suppressNextRestoredAudioRef = useRef(false);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // Initialize audio context
  const initAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000,
        latencyHint: 'interactive'
      });
      pcmSchedulerRef.current = createPcmStreamScheduler(audioContextRef.current, {
        primingMs: 160,
        onPlaybackStart: () => {
          const scheduler = pcmSchedulerRef.current;
          if (!scheduler) return;
          streamedAudioSinceCutRef.current = true;
          speechStartTimeRef.current = audioContextRef.current?.currentTime || 0;
          nextStartTimeRef.current = scheduler.nextStartTime;
          speechTotalDurationRef.current = Math.max(
            0,
            scheduler.nextStartTime - speechStartTimeRef.current
          );
          setIsAISpeaking(true);
          setIsWaitingForAIResponse(false);
        },
        onPlaybackIdle: () => {
          setIsAISpeaking(false);
          setIsWaitingForAIResponse(false);
        },
      });
    }
    if (audioContextRef.current.state === 'suspended') {
      return audioContextRef.current.resume().catch(error => {
        console.warn('AudioContext resume rejected:', error?.message || error);
        throw error;
      });
    }
    return Promise.resolve();
  };

  // Stop audio playback
  const stopAudioPlayback = () => {
    isInterruptedRef.current = true;
    pcmSchedulerRef.current?.stop();
    pendingStreamAudioRef.current = [];
    receivedStreamAudioRef.current = false;
    aiTextReadyForAudioRef.current = false;
    activeAudioResponseIdRef.current = null;
    streamAudioDoneRef.current = false;
    audioQueueRef.current.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // Ignore errors from already stopped sources
      }
    });
    audioQueueRef.current = [];
    // Also stop any progressive HTMLAudio replay in flight.
    if (htmlAudioRef.current) {
      try { htmlAudioRef.current.pause(); } catch (e) { /* ignore */ }
      htmlAudioRef.current = null;
    }
    nextStartTimeRef.current = 0;
    // New turn boundary: forget whether the previous turn streamed audio.
    streamedAudioSinceCutRef.current = false;
    setIsAISpeaking(false);
    setPlayingAudioUrl(null);
  };

  // Play full audio (for AudioBar) - use proxy for cross-origin audio
  // autoQueue=false (default): interrupt current audio and play immediately (audio_back replay)
  // autoQueue=true: schedule after current audio without interruption (auto-play on new AI message)
  const playFullAudio = (audioUrl, autoQueue = false) => {
    console.log('Playing full audio from:', audioUrl, autoQueue ? '(queued)' : '(interrupt)');

    if (!autoQueue) {
      // User-triggered replay: stop current audio and play immediately
      stopAudioPlayback();
      isInterruptedRef.current = false;
    }

    // Check if URL is cross-origin
    const isCrossOrigin = audioUrl.startsWith('http') && !audioUrl.startsWith(window.location.origin);

    if (isCrossOrigin) {
      // For cross-origin audio, always use proxy to avoid CORS issues
      console.log('Using proxy for cross-origin audio:', audioUrl);
      fetchAudioViaProxy(audioUrl, autoQueue);
    } else {
      // Same-origin, use Web Audio API
      initAudioContext();
      fetch(audioUrl)
        .then(res => res.arrayBuffer())
        .then(buffer => {
          if (!audioContextRef.current) return;
          return audioContextRef.current.decodeAudioData(buffer);
        })
        .then(audioBuffer => {
          if (!audioBuffer || !audioContextRef.current) return;
          const ctx = audioContextRef.current;
          const source = ctx.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(ctx.destination);

          // Add to queue for stop functionality
          audioQueueRef.current.push(source);

          // Reset time drift if autoQueue has drifted more than 30 seconds
          const TIME_DRIFT_THRESHOLD = 30;
          if (autoQueue && nextStartTimeRef.current - ctx.currentTime > TIME_DRIFT_THRESHOLD) {
            console.warn(`Auto-queue time drift detected (${(nextStartTimeRef.current - ctx.currentTime).toFixed(2)}s). Resetting to current time.`);
            nextStartTimeRef.current = ctx.currentTime;
          }

          const isFirstChunkOfTurn = audioQueueRef.current.length === 1; // already pushed above
          const start = autoQueue ? Math.max(ctx.currentTime, nextStartTimeRef.current) : ctx.currentTime;
          source.start(start);
          if (autoQueue) nextStartTimeRef.current = start + audioBuffer.duration;
          if (isFirstChunkOfTurn) speechStartTimeRef.current = start;
          speechTotalDurationRef.current = (autoQueue ? nextStartTimeRef.current : start + audioBuffer.duration) - speechStartTimeRef.current;

          setIsAISpeaking(true);
          setIsWaitingForAIResponse(false);
          setPlayingAudioUrl(audioUrl);
          source.onended = () => {
            // Remove this source from the queue, then only clear speaking
            // state when no other source is still scheduled. Mirrors the
            // streaming-PCM path so CC caption visibility stays consistent.
            audioQueueRef.current = audioQueueRef.current.filter(s => s !== source);
            setPlayingAudioUrl(prev => prev === audioUrl ? null : prev);
            const ctxNow = audioContextRef.current?.currentTime ?? 0;
            if (audioQueueRef.current.length === 0 && nextStartTimeRef.current <= ctxNow + 0.05) {
              setIsAISpeaking(false);
              // Turn finished — ensure the mascot's thinking face can't persist.
              setIsWaitingForAIResponse(false);
            }
            console.log('Audio playback ended');
          };
        })
        .catch(err => console.error('Error playing same-origin audio:', err));
    }
  };

  // Fetch audio via API proxy to avoid CORS issues.
  // Two playback strategies (see shouldUseProgressiveAudio):
  //  - Progressive HTMLAudioElement: edge-download-and-play, first byte out
  //    immediately. Used for user replay (autoQueue=false) — the common case —
  //    where there's no sample-accurate queue to preserve. On the production
  //    path (CN browser → Cloudflare → Bangkok Nginx → COS Shanghai) the old
  //    fetch-whole-WAV-then-decode added ~3s of silence before first sound.
  //  - Web Audio buffer: retained for autoQueue=true chaining, which needs
  //    Web Audio's sample-level scheduling that HTMLAudio can't provide.
  const fetchAudioViaProxy = async (audioUrl, autoQueue = false) => {
    const proxyUrl = `/api/media/proxy?url=${encodeURIComponent(audioUrl)}`;

    if (shouldUseProgressiveAudio(autoQueue, nextStartTimeRef.current, audioContextRef.current?.currentTime ?? 0)) {
      // Try the raw COS URL directly first (fast, ~200ms from CN, Range-capable,
      // no proxy handshake). On error, fall back to the media proxy once. A
      // proxy failure is terminal — nextProgressiveAttempt returns null to
      // prevent a direct↔proxy retry loop.
      const startProgressive = (attempt) => {
        try {
          const src = progressiveAudioSrc(audioUrl, attempt);
          const audio = new Audio(src);
          // Replace any previous progressive element so stop/overlap is clean.
          if (htmlAudioRef.current) {
            try { htmlAudioRef.current.pause(); } catch (_) {}
          }
          htmlAudioRef.current = audio;

          setIsAISpeaking(true);
          setIsWaitingForAIResponse(false);
          setPlayingAudioUrl(audioUrl);

          // Feed the CC caption progress ratio from the element itself. Once
          // metadata is known we anchor speechStartTimeRef to the AudioContext
          // clock so getProgressRatio (ctx.currentTime - start)/total still works.
          audio.addEventListener('loadedmetadata', () => {
            const ctx = audioContextRef.current;
            if (ctx && isFinite(audio.duration)) {
              speechStartTimeRef.current = ctx.currentTime;
              speechTotalDurationRef.current = audio.duration;
            }
          });

          const cleanup = () => {
            if (htmlAudioRef.current === audio) htmlAudioRef.current = null;
            setPlayingAudioUrl(prev => prev === audioUrl ? null : prev);
            // Only clear speaking state if no Web Audio queue is still running.
            const ctxNow = audioContextRef.current?.currentTime ?? 0;
            if (audioQueueRef.current.length === 0 && nextStartTimeRef.current <= ctxNow + 0.05) {
              setIsAISpeaking(false);
              setIsWaitingForAIResponse(false);
            }
          };
          audio.addEventListener('ended', cleanup);
          audio.addEventListener('pause', cleanup);
          audio.addEventListener('error', () => {
            // Only this element still current? Otherwise a newer play superseded it.
            if (htmlAudioRef.current !== audio) return;
            const fallback = nextProgressiveAttempt(attempt);
            if (fallback) {
              try { audio.pause(); } catch (_) {}
              startProgressive(fallback);
            } else {
              cleanup();
            }
          });

          audio.play().catch(() => {
            // play() rejection (autoplay/decode) — try the fallback path too.
            if (htmlAudioRef.current !== audio) return;
            const fallback = nextProgressiveAttempt(attempt);
            if (fallback) startProgressive(fallback);
          });
        } catch (err) {
          // Silently ignore playback errors - audio playback is optional
        }
      };

      startProgressive('direct');
      return;
    }

    try {
      const response = await fetch(proxyUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const arrayBuffer = await response.arrayBuffer();

      initAudioContext();
      if (!audioContextRef.current) return;

      const ctx = audioContextRef.current;
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      // Add to queue for stop functionality
      audioQueueRef.current.push(source);

      // Reset time drift if autoQueue has drifted more than 30 seconds
      const TIME_DRIFT_THRESHOLD = 30;
      if (autoQueue && nextStartTimeRef.current - ctx.currentTime > TIME_DRIFT_THRESHOLD) {
        console.warn(`Auto-queue time drift detected (${(nextStartTimeRef.current - ctx.currentTime).toFixed(2)}s). Resetting to current time.`);
        nextStartTimeRef.current = ctx.currentTime;
      }

      const isFirstChunkOfTurn = audioQueueRef.current.length === 1; // already pushed above
      const start = autoQueue ? Math.max(ctx.currentTime, nextStartTimeRef.current) : ctx.currentTime;
      source.start(start);
      if (autoQueue) nextStartTimeRef.current = start + audioBuffer.duration;
      if (isFirstChunkOfTurn) speechStartTimeRef.current = start;
      speechTotalDurationRef.current = (autoQueue ? nextStartTimeRef.current : start + audioBuffer.duration) - speechStartTimeRef.current;

      setIsAISpeaking(true);
      setPlayingAudioUrl(audioUrl);
      source.onended = () => {
        audioQueueRef.current = audioQueueRef.current.filter(s => s !== source);
        setPlayingAudioUrl(prev => prev === audioUrl ? null : prev);
        const ctxNow = audioContextRef.current?.currentTime ?? 0;
        if (audioQueueRef.current.length === 0 && nextStartTimeRef.current <= ctxNow + 0.05) {
          setIsAISpeaking(false);
          setIsWaitingForAIResponse(false);
        }
      };
    } catch (err) {
      // Silently ignore proxy errors - audio playback is optional
    }
  };

  // Text-to-speech for selected text
  const playSelectedText = async () => {
    if (!selection.text || isSynthesizing) return;
    
    setIsSynthesizing(true);
    try {
      const blob = await aiAPI.tts(selection.text);
      const audioUrl = URL.createObjectURL(blob);
      playFullAudio(audioUrl);
    } catch (error) {
      console.error('Speech synthesis error:', error);
    } finally {
      setIsSynthesizing(false);
      setSelection(prev => ({ ...prev, visible: false }));
    }
  };

  // Handle text selection
  const handleTextSelection = () => {
    const selectionObj = window.getSelection();
    const selectedText = selectionObj.toString().trim();
    
    if (selectedText.length > 0) {
      const range = selectionObj.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      
      setSelection({
        text: selectedText,
        x: Math.min(Math.max(rect.left + rect.width / 2, 50), window.innerWidth - 50),
        y: Math.max(rect.top - 50, 50),
        visible: true
      });
    } else {
      setSelection(prev => ({ ...prev, visible: false }));
    }
  };

  // Handle next scenario
  const handleNextScenario = () => {
    const nextIndex = currentScenarioIndex + 1;
    if (nextIndex < allScenarios.length) {
      const nextScenario = allScenarios[nextIndex];
      navigate('/conversation', { 
        state: { 
          scenario: nextScenario.title, 
          tasks: nextScenario.tasks,
          allScenarios: allScenarios,
          currentIndex: nextIndex
        } 
      });
    }
  };

  // Handle retry current scenario
  // Options: { keepHistory: boolean, resetProgress: boolean }
  // - keepHistory: 是否保留对话历史
  // - resetProgress: 是否重置进度（true=重新开始，false=继续练习）
  const handleRetryCurrentScenario = async (options = {}) => {
    const { keepHistory = true, resetProgress = false } = options;
    
    // Get scenario from URL params or state
    const searchParams = new URLSearchParams(window.location.search);
    const scenarioFromUrl = searchParams.get('scenario');
    const scenarioFromState = location.state?.scenario;
    const scenarioTitle = scenarioFromState || scenarioFromUrl;

    console.log('Retrying scenario:', scenarioTitle, 'Keep history:', keepHistory, 'Reset progress:', resetProgress);

    // Only reset tasks if user explicitly wants to start over
    if (resetProgress) {
      try {
        if (scenarioTitle) {
          console.log('Resetting all tasks in scenario:', scenarioTitle);
          const resetResult = await userAPI.resetTask(null, scenarioTitle);
          scoringGenerationByTaskRef.current = new Map(
            (resetResult?.tasks || []).map(task => [
              String(task.task_id),
              Number(task.scoring_generation),
            ])
          );
        }
      } catch (err) {
        console.error('Failed to reset scenario:', err);
        setMessages(prev => [...prev, {
          type: 'system',
          content: '重置失败，当前进度已保留，请重试。',
          isFinal: true,
        }]);
        return;
      }
    }

    setShowCompletionModal(false);
    // Keep completedTasks if not resetting progress
    if (resetProgress) {
      setCompletedTasks(new Set());
    }
    completionCheckedRef.current = resetProgress ? false : true;
    // Keep modal view tracking to prevent re-showing on refresh
    hasViewedCompletionModalRef.current = true;
    // Keep progress at 100% if not resetting, otherwise reset to 0
    if (resetProgress) {
      setCurrentTaskProgress(0);
      setCurrentTaskScore(0);
      setProgressFeedback(null);
      setTaskReadyToComplete(null);
      setTaskCompletionPending(false);
      feedbackOrderRef.current.clear();
      // 重置魔法重复阶段状态
      setMagicPassedTasks(new Set());
      setCurrentMagicSentence('');
      setMagicCardState('waiting');
      setMagicCardCovered(false);
      setCurrentPhase(isRecallMode ? 'magic_repetition' : 'scene_theater');
      // 清除 localStorage 里的魔法通过记录
      const scKey = scenarioFromUrl || '';
      if (scKey) localStorage.removeItem(_lsScenarioKey('magic_passed_', scKey));
      if (scKey) localStorage.removeItem(_lsScenarioKey('magic_sentence_', scKey));
    } else {
      // Continue practice: show 100% progress
      setCurrentTaskProgress(100);
      setCurrentTaskScore(9); // Max score for completed tasks
    }
    previousProgressRef.current = resetProgress ? 0 : 100; // Reset progress tracking
    lastProficiencyUpdateRef.current = null; // Reset deduplication

    // Optionally clear messages (default: keep history)
    if (!keepHistory) {
      setMessages([
        {
          type: 'system',
          content: '重新开始练习当前场景...'
        }
      ]);
    }

    // Refresh tasks from backend to get updated status
    try {
      const updatedGoal = await userAPI.getActiveGoal();
      if (updatedGoal && updatedGoal.goal && updatedGoal.goal.scenarios) {
        const matchedScenario = updatedGoal.goal.scenarios.find(
          s => s.title === scenarioTitle ||
               (scenarioTitle && s.title.toLowerCase().includes(scenarioTitle.toLowerCase()))
        );
        if (matchedScenario) {
          setTasks(matchedScenario.tasks);
          console.log('Tasks refreshed after retry:', matchedScenario.tasks);
        }
      }
    } catch (err) {
      console.error('Failed to refresh tasks:', err);
    }

    // Reset session to clear AI context and restart with first task prompt
    // Only reset session when user wants to start over
    if (resetProgress) {
      try {
        if (sessionId) {
          // Properly cleanup WebSocket connection
          if (socketRef.current) {
            socketRef.current.removeAllListeners();
            socketRef.current.destroy();
            socketRef.current = null;
          }

          // Clear old session from sessionStorage and localStorage
          sessionStorage.removeItem('session_id');

          // Clear scenario-specific session from localStorage to prevent history reload on refresh
          if (scenarioTitle) {
            if (user?.id) {
              localStorage.removeItem(_lsSessionKey(user.id, scenarioTitle));
            }
            localStorage.removeItem(_lsScenarioKey('session_', scenarioTitle));
            console.log('Cleared localStorage session for scenario:', scenarioTitle);
          }

          setSessionId(null);

          // Create new session which will trigger AI to use first task prompt
          const newSessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
          sessionStorage.setItem('session_id', newSessionId);
          setSessionId(newSessionId);

          console.log('New session created for retry:', newSessionId);
        }
      } catch (err) {
        console.error('Failed to reset session:', err);
      }
    }

    // Clear localStorage for this scenario to prevent old progress restoration
    if (resetProgress && scenarioTitle) {
      localStorage.removeItem(_lsScenarioKey('task_progress_', scenarioTitle));
      localStorage.removeItem(_lsScenarioKey('welcome_muted_', scenarioTitle));
      console.log('Cleared localStorage progress for scenario:', scenarioTitle);
    }

    // Only refresh page when resetting progress (to establish new WebSocket connection)
    // For "continue practice", just close the modal and keep current state
    if (resetProgress) {
      console.log('Refreshing page to establish new connection...');
      window.location.reload();
    }
  };

  // Handle select other scenario
  const handleSelectOtherScenario = () => {
    navigate('/discovery');
  };

  // Handle back to discovery
  const handleBackToDiscovery = () => {
    navigate('/discovery');
  };

  // Handle task completion confirmation
  const handleConfirmComplete = () => {
    if (!taskReadyToComplete || !acceptScoringMessage(taskReadyToComplete) || taskCompletionPending) return;
    const wsReadyState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
    if (wsReadyState === WebSocket.OPEN) {
      console.log('🏁 Sending user_confirmed_complete:', taskReadyToComplete.task_id);
      socketRef.current.send(JSON.stringify({
        type: 'user_confirmed_complete',
        payload: {
          task_id: taskReadyToComplete.task_id,
          ready_token: taskReadyToComplete.ready_token,
        },
      }));
      setTaskCompletionPending(true);
    } else {
      console.error('❌ WebSocket not open, cannot send confirmation');
    }
  };

  // Manual retry reconnect function
  const handleManualRetry = useCallback(() => {
    console.log('Manual retry triggered');
    if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
    setIsManualDisconnect(false);
    isManualDisconnectRef.current = false;
    setReconnectAttempts(0);
    reconnectAttemptsRef.current = 0;
    setWebSocketError(null);
    wsRejectedRef.current = false;
    setWsRejected(false);

    // Reconnect with current session ID
    if (sessionId) {
      // Use latest connectWebSocket reference
      connectWebSocketRef.current(sessionId);
    }
  }, [sessionId]);

  // Save conversation history
  const saveConversationHistory = async (sessionIdOverride = null, messagesOverride = null, options = {}) => {
    const activeSessionId = sessionIdOverride || sessionIdRef.current || sessionId;
    const activeMessages = messagesOverride || messages;
    if (!activeSessionId || activeMessages.length === 0) {
      console.log('No session ID or messages to save');
      return;
    }

    try {
      console.log('Saving conversation history. Total messages:', activeMessages.length);
      console.log('Messages before filtering:', activeMessages.map((m, i) => ({index: i, type: m.type, isFinal: m.isFinal, content: m.content?.substring(0, 50)})));
      
      // Prepare messages for saving - save final messages and non-final AI messages
      const messagesToSave = prepareHistorySnapshot(activeMessages);

      console.log('Messages after filtering:', messagesToSave.map((m, i) => ({index: i, role: m.role, content: m.content?.substring(0, 50)})));

      if (messagesToSave.length === 0) {
        console.log('No finalized messages to save');
        return;
      }

      const response = await conversationAPI.saveHistory(activeSessionId, messagesToSave, user.id, options);
      if (response?.success) {
        console.log('Conversation history saved successfully');
      } else {
        console.error('Failed to save conversation history:', response.message);
      }
    } catch (error) {
      console.error('Error saving conversation history:', error);
    }
  };

  const connectionToastShownRef = useRef(false);

  const releasePendingAudioAfterPaint = (responseId) => {
    activeAudioResponseIdRef.current = responseId || null;
    aiTextReadyForAudioRef.current = true;
    const schedule = window.requestAnimationFrame || ((callback) => setTimeout(callback, 0));
    schedule(() => {
      const pending = pendingStreamAudioRef.current.splice(0);
      pending.forEach(packet => {
        if (packet.packetPromise) {
          playAudioChunkRef.current?.(packet.packetPromise.then(resolved => (
            !resolved.responseId || resolved.responseId === activeAudioResponseIdRef.current
              ? resolved.pcm
              : new Uint8Array()
          )));
        } else if (!packet.responseId || packet.responseId === activeAudioResponseIdRef.current) {
          playAudioChunkRef.current?.(packet.pcm);
        }
      });
      if (
        streamAudioDoneRef.current === true
        || streamAudioDoneRef.current === activeAudioResponseIdRef.current
      ) {
        const scheduler = pcmSchedulerRef.current;
        scheduler?.flush(scheduler.generation);
      }
    });
  };

  // Handle JSON messages from WebSocket
  const handleJsonMessage = useCallback((data) => {
      console.log('Received JSON message:', data);

      // Handle different message types
      switch (data.type) {
        case 'session_restored': {
           const restored = data.payload || {};
           const restoredScore = Number(restored.score || 0);
           const restoredCount = Number(restored.interaction_count || 0);
           const restoredProgress = calculateTaskProgress({
             score: restoredScore,
             interactionCount: restoredCount,
             taskCompleted: Boolean(restored.task_completed),
           });
           if (acceptScoringMessage(restored)) {
             setTaskReadyToComplete(null);
             setTaskCompletionPending(false);
             feedbackOrderRef.current.set(`${restored.task_id}:${restored.scoring_generation ?? 0}`, restoredCount);
             setCurrentTaskScore(restoredScore);
             setCurrentTaskProgress(restoredProgress);
             setProgressFeedback(restored.progress_feedback || null);
             previousProgressRef.current = restoredProgress;
             if (restored.task_id) lastSeenTaskIdRef.current = restored.task_id;
           }
           isRestoringSessionRef.current = false;
           setIsRestoringSession(false);
           setIsConnected(true);
           setWebSocketError(null);
           console.log('Session restored:', restored);
           break;
        }
        case 'connection_established':
           console.log('Connection established:', data.payload);
           // Only show toast once per page session to avoid spam
           if (connectionToastShownRef.current) {
               console.log('Connection toast already shown, skipping');
               break;
           }
           connectionToastShownRef.current = true;

           // Don't show connection message in chat - it's handled by UI status indicator
           // Just log the connection for debugging
           console.log('WebSocket connected, role:', data.payload?.role);

           // Re-fetch tasks after connection to ensure we have the latest state
           // This is especially important after page refresh
           const searchParams = new URLSearchParams(window.location.search);
           const scenario = searchParams.get('scenario') || location.state?.scenario;
           if (scenario) {
               // Re-fetch the latest goal state from DB to sync task progress
               userAPI.getActiveGoal().then(res => {
                   if (res && res.goal && res.goal.scenarios) {
                       let activeScenario = res.goal.scenarios.find(s => s.title.trim() === scenario.trim());
                       
                       // Try case-insensitive match if exact match fails
                       if (!activeScenario) {
                           activeScenario = res.goal.scenarios.find(s => 
                               s.title.toLowerCase() === scenario.toLowerCase()
                           );
                       }
                       
                       // Try partial match as fallback
                       if (!activeScenario) {
                           activeScenario = res.goal.scenarios.find(s =>
                               s.title.toLowerCase().includes(scenario.toLowerCase()) ||
                               scenario.toLowerCase().includes(s.title.toLowerCase())
                           );
                       }
                       
                       if (activeScenario && activeScenario.tasks) {
                           setTasks(activeScenario.tasks);
                           console.log('Updated tasks from backend:', activeScenario.tasks);

                           // Re-calculate completed set and current task progress
                           const newCompleted = new Set();
                           let currentTaskProgress = 0;
                           let currentTaskScore = 0;

                           activeScenario.tasks.forEach(t => {
                               if (typeof t === 'object') {
                                   if (t.status === 'completed') {
                                       newCompleted.add(t.text);
                                   } else if (t.status === 'pending' || t.status === 'in_progress') {
                                       // Get progress from the first incomplete task
                                       if (currentTaskProgress === 0) {
                                           currentTaskScore = Number(t.score || 0);
                                           const interactionCount = Number(t.interaction_count || 0);
                                           currentTaskProgress = calculateTaskProgress({
                                             score: currentTaskScore,
                                             interactionCount,
                                             taskCompleted: false,
                                           });
                                       }
                                   }
                               }
                           });
                           setCompletedTasks(newCompleted);

                           // Force sync progress bar to backend true value (even if 0)
                           setCurrentTaskProgress(currentTaskProgress);
                           setCurrentTaskScore(currentTaskScore);
                           previousProgressRef.current = currentTaskProgress;
                           localStorage.setItem(_lsScenarioKey('task_progress_', scenario), currentTaskProgress.toString());

                           // Show toast for newly completed task
                           const completedTask = activeScenario.tasks.find(t => t.status === 'completed' && !newCompleted.has(t.text));
                           if (completedTask) { setMessages(prev => [...prev, { type: 'system', content: `✅ 完成任务：${completedTask.text}` }]); }
                       }
                   }
               }).catch(err => console.error('Failed to sync tasks:', err));
           }
           break;
        case 'connection_closed':
           // The upstream model socket is gone even if the browser transport
           // has not emitted close yet. Enter restoration state immediately;
           // comms will close this socket with a retryable code exactly once.
           setIsConnected(false);
           isRestoringSessionRef.current = true;
           setIsRestoringSession(true);
           setIsWaitingForAIResponse(false);
           // DashScope can close while the AI service WebSocket itself remains
           // open. Force the browser transport through its normal retryable
           // close handler instead of waiting until the next user utterance.
           {
             const affectedSocket = socketRef.current;
             const reconnectable = data.payload?.reconnectable !== false;
             if (!reconnectable) {
               isRestoringSessionRef.current = false;
               setIsRestoringSession(false);
               wsRejectedRef.current = true;
               setWsRejected(true);
               setWebSocketError(data.payload?.reason || 'AI 服务拒绝连接');
             }
             setTimeout(() => {
               if (
                 socketRef.current === affectedSocket
                 && affectedSocket?.getReadyState?.() === WebSocket.OPEN
               ) {
                 affectedSocket.close(
                   reconnectable ? 4002 : 4400,
                   reconnectable ? 'Upstream model disconnected' : 'Upstream rejected connection'
                 );
               }
             }, 0);
           }
           break;
        case 'transcription':
           console.log('Transcription Event:', data);
           restoredAiContentKeysRef.current.clear();
           // User transcription
           setMessages(prev => {
               const last = prev[prev.length - 1];
               const currentId = currentUserMessageIdRef.current;

               // STRICT CHECK: Update ONLY if the last message matches the current turn ID
               if (last && last.type === 'user' && last.id === currentId && !last.isFinal) {
                   const updated = [
                       ...prev.slice(0, -1),
                       {
                           ...last,
                           content: data.isFinal ? data.text : last.content + data.text,
                           isFinal: !!data.isFinal
                       }
                   ];
                   console.log('Updated existing user message:', updated[updated.length - 1]);
                   return updated;
               }

               // Otherwise, append a NEW message for this turn
               // This prevents overwriting previous turns if they weren't finalized correctly
               const newMessage = {
                   type: 'user',
                   content: data.text,
                   isFinal: !!data.isFinal,
                   id: currentId // Bind this message to the current turn
               };
               console.log('Created new user message:', newMessage);
               return [...prev, newMessage];
           });
           break;
        case 'audio_url':
           const audioPayload = data.payload || data;
           const url = audioPayload.url || data.url;
           const role = audioPayload.role || data.role;
           const targetResponseId = data.responseId || audioPayload.responseId; // Get ID from event

           // Check if welcome message is muted (after retry)
           const currentScenario = new URLSearchParams(window.location.search).get('scenario');
           const welcomeMuted = currentScenario ? localStorage.getItem(_lsScenarioKey('welcome_muted_', currentScenario)) === 'true' : false;

           // De-dupe: if streaming PCM already played audible audio for this turn,
           // suppress the COS auto-play (it would be a second playback). The COS
           // URL is still attached so the user can manually replay; we only set
           // audioPlayed=true to keep the auto-play useEffect from firing.
           // Consume the flag here (read once per audio_url) so a later distinct
           // turn that did NOT stream isn't accidentally suppressed.
           const streamedThisTurn = streamedAudioSinceCutRef.current;

           if (role === 'assistant') {
               if (suppressNextRestoredAudioRef.current) {
                   suppressNextRestoredAudioRef.current = false;
                   break;
               }
               setMessages(prev => {
                   const newMessages = [...prev];

                   // 1. Try Strict Match by Response ID
                   if (targetResponseId) {
                       const index = newMessages.findIndex(m => m.type === 'ai' && m.responseId === targetResponseId);
                       if (index !== -1) {
                           console.log(`[AudioURL] Attached to message ${index} via ID ${targetResponseId}, isFinal=${newMessages[index].isFinal}, streamed=${streamedThisTurn}`);
                           newMessages[index] = {
                               ...newMessages[index],
                               audioUrl: url,
                               audioPlayed: streamedThisTurn || shouldSuppressAutoPlay(welcomeMuted, newMessages, index)
                           };
                           return newMessages;
                       }
                   }

                   // 2. Fallback: Attach to the LAST AI message that doesn't have a URL
                   for (let i = newMessages.length - 1; i >= 0; i--) {
                       if (newMessages[i].type === 'ai' && !newMessages[i].audioUrl) {
                           console.log(`[AudioURL] Fallback attachment to message ${i}, isFinal=${newMessages[i].isFinal}, streamed=${streamedThisTurn}`);
                           newMessages[i] = {
                               ...newMessages[i],
                               audioUrl: url,
                               audioPlayed: streamedThisTurn || shouldSuppressAutoPlay(welcomeMuted, newMessages, i)
                           };
                           break;
                       }
                   }
                   return newMessages;
               });
           } else if (role === 'user') {
               setMessages(prev => {
                   const newMessages = [...prev];
                   const currentId = currentUserMessageIdRef.current;
                   // Attach URL ONLY to the message with the matching ID
                   for (let i = newMessages.length - 1; i >= 0; i--) {
                       if (newMessages[i].type === 'user' && newMessages[i].id === currentId) {
                           newMessages[i] = { ...newMessages[i], audioUrl: url };
                           console.log(`[AudioURL] Attached to user message with ID ${currentId}`);
                           break;
                       }
                   }
                   return newMessages;
               });
           }
           break;
        case 'role_switch':
           setCurrentRole(data.payload.role);
           console.log('Role switched to:', data.payload.role);
           break;
        case 'ai_text_delta': {
           // Streaming AI reply text — arrives ~4s before the one-shot
           // `ai_message`/`ai_response`, roughly in sync with streaming audio.
           // Append the raw delta to the in-progress AI message and display the
           // marker-cleaned view. `streamRaw` holds the un-cleaned accumulation
           // so a marker split across deltas (e.g. "[TASK_" then "1_COMPLETE]")
           // re-closes correctly instead of staying hidden.
           const deltaPayload = data.payload || data;
           const delta = deltaPayload.delta || '';
           const deltaResponseId = deltaPayload.responseId;
           if (!delta) break;

           // Commit the transcript bubble before releasing buffered PCM. The
           // scheduler's priming window then gives React time to paint it.
           releasePendingAudioAfterPaint(deltaResponseId);

           // A text delta means the model has started replying — drop the
           // thinking/ellipsis state so streamed text/audio can show.
           setIsWaitingForAIResponse(false);

           setMessages(prev => {
               const last = prev[prev.length - 1];
               if (last && last.type === 'ai' && !last.isFinal) {
                   const rawNext = appendDelta(last.streamRaw ?? last.content, delta);
                   return [
                       ...prev.slice(0, -1),
                       {
                           ...last,
                           streamRaw: rawNext,
                           content: cleanStreamingText(rawNext),
                           responseId: last.responseId || deltaResponseId
                       }
                   ];
               }
               const rawNext = appendDelta('', delta);
               return [...prev, {
                   type: 'ai',
                   streamRaw: rawNext,
                   content: cleanStreamingText(rawNext),
                   isFinal: false,
                   responseId: deltaResponseId
               }];
           });
           break;
        }
        case 'ai_message':
           // Handle AI message from comms-service (contains text content in payload)
           console.log('🤖 AI Message:', data);
           const msgPayload = data.payload || data;
           const aiContent = msgPayload.content || data.content || data.text || msgPayload.text || '';
           const responseId = msgPayload.responseId || data.responseId;
           const responseTurnId = msgPayload.turn_id || data.turn_id;
           releasePendingAudioAfterPaint(responseId);

           // 检测文本标记（降级方案）
           let cleanContent = aiContent;

           // 提取 MAGIC_SENTENCE（兼容 [ ] 和 < > 两种括号，兼容缺少结尾符/嵌套括号）
           if (aiContent && (aiContent.includes('[MAGIC_SENTENCE:') || aiContent.includes('<MAGIC_SENTENCE:'))) {
               const raw = extractMagicSentence(aiContent);
               if (raw) {
                   // 去掉末尾的 "Please repeat..." 等指令文字
                   const sentence = raw
                       .replace(/\.?\s*[Pp]lease\s+(repeat|say|try|recite)[\s\S]*/i, '')
                       .trim();
                   setCurrentMagicSentence(sentence || raw);
               }
               // 从显示文字中移除标记（共享正则，兼容嵌套括号）
               cleanContent = stripAllMarkers(aiContent).trim();
           }

           if (aiContent && aiContent.includes('[MAGIC_PASS]')) {
               // 从显示文字中移除标记
               cleanContent = cleanContent.replace(/\s*\[MAGIC_PASS[^\]]*\]/g, '');
               // 触发台词卡"通过"动画：仅在背诵模式下才生效（防止跟读阶段误触发）
               if (magicCardState === 'reciting') {
                   setMagicCardCovered(false);  // 背诵模式立即揭开
                   setMagicCardState('passed');
                   setTimeout(() => {
                       setMagicCardState('waiting');
                   }, 1800);
               }
           }
           if (aiContent && /\[TASK_\d+_COMPLETE\]/.test(aiContent)) {
               // 提取任务索引（假设格式: [TASK_0_COMPLETE])
               const match = aiContent.match(/\[TASK_(\d+)_COMPLETE\]/);
               if (match) {
                   const taskIdx = parseInt(match[1], 10);
                   setTheaterCompletedTasks(prev => new Set([...prev, taskIdx]));
               }
           }

           if (cleanContent) {
               const restoredKey = historyContentKey({ type: 'ai', content: cleanContent });
               if (restoredAiContentKeysRef.current.delete(restoredKey)) {
                   suppressNextRestoredAudioRef.current = true;
                   setMessages(prev => {
                       const last = prev[prev.length - 1];
                       return last?.type === 'ai' && !last.isFinal ? prev.slice(0, -1) : prev;
                   });
                   break;
               }
               setMessages(prev => {
                   const last = prev[prev.length - 1];
                   // If last message is an in-progress AI message, update it
                   if (last && last.type === 'ai' && !last.isFinal) {
                       return [
                           ...prev.slice(0, -1),
                           {
                               ...last,
                               content: cleanContent,
                               isFinal: true,
                               responseId: responseId || last.responseId,
                               turn_id: responseTurnId || last.turn_id
                           }
                       ];
                   }
                   // Otherwise create new AI message
                   return [...prev, {
                       type: 'ai',
                       content: cleanContent,
                       isFinal: true,
                       responseId: responseId,
                       turn_id: responseTurnId
                   }];
               });
           } else {
               // Pure-marker reply (e.g. only [MAGIC_PASS]) strips to empty.
               // The streaming ai_text_delta path may have already created an
               // in-progress AI bubble — without finalizing it, it spins forever
               // (render state ties loading to !isFinal). An empty bubble has no
               // display value, so drop it rather than leave a blank loading row.
               setMessages(prev => {
                   const last = prev[prev.length - 1];
                   if (last && last.type === 'ai' && !last.isFinal) {
                       return prev.slice(0, -1);
                   }
                   return prev;
               });
           }
           break;
        case 'ai_turn_started': {
           const startedResponseId = data.payload?.responseId || data.responseId;
           setMessages(prev => {
             const last = prev[prev.length - 1];
             if (last?.type === 'ai' && !last.isFinal) return prev;
             return [...prev, {
               type: 'ai', content: '', isFinal: false, responseId: startedResponseId,
             }];
           });
           releasePendingAudioAfterPaint(startedResponseId);
           break;
        }
        case 'ai_response': {
           // Handle AI text response from comms-service
           let responseText = data.text || '';
           console.log('🤖 AI Response:', responseText);

           // 提取 MAGIC_SENTENCE 标记（兼容 [ ] 和 < > 两种括号，兼容缺少结尾符/嵌套括号）
           if (responseText.includes('[MAGIC_SENTENCE:') || responseText.includes('<MAGIC_SENTENCE:')) {
               const raw = extractMagicSentence(responseText);
               if (raw) {
                   const sentence = raw
                       .replace(/\.?\s*[Pp]lease\s+(repeat|say|try|recite)[\s\S]*/i, '')
                       .trim();
                   setCurrentMagicSentence(sentence || raw);
               }
               responseText = stripAllMarkers(responseText);
           }

           // 提取并移除 MAGIC_PASS 标记
           if (responseText.includes('[MAGIC_PASS]')) {
               responseText = responseText.replace(/\s*\[MAGIC_PASS[^\]]*\]/g, '');
               // 仅在背诵模式下才触发通过动画（防止跟读阶段误触发）
               if (magicCardState === 'reciting') {
                   setMagicCardCovered(false);
                   setMagicCardState('passed');
                   setTimeout(() => {
                       setMagicCardState('waiting');
                   }, 1800);
               }
           }

           const finalText = responseText.trim();
           const restoredKey = historyContentKey({ type: 'ai', content: finalText });
           if (finalText && restoredAiContentKeysRef.current.delete(restoredKey)) {
               suppressNextRestoredAudioRef.current = true;
               setMessages(prev => {
                   const last = prev[prev.length - 1];
                   return last?.type === 'ai' && !last.isFinal ? prev.slice(0, -1) : prev;
               });
               break;
           }
           setMessages(prev => {
               const last = prev[prev.length - 1];
               if (last && last.type === 'ai' && !last.isFinal) {
                   // Pure-marker reply strips to empty: drop the in-progress
                   // (streaming) bubble instead of leaving it spinning forever.
                   if (!finalText) {
                       return prev.slice(0, -1);
                   }
                   return [...prev.slice(0, -1), { ...last, content: finalText, isFinal: true }];
               }
               // No in-progress bubble to finalize; only append when non-empty.
               if (!finalText) {
                   return prev;
               }
               return [...prev, { type: 'ai', content: finalText, isFinal: true }];
           });
           break;
        }
        case 'user_transcript':
           // Display user's speech transcription in chat
           if (data.payload && data.payload.text) {
             restoredAiContentKeysRef.current.clear();
             setMessages(prev => reconcileUserTranscript(prev, {
               text: data.payload.text,
               messageId: data.payload.messageId || data.payload.message_id,
               currentMessageId: currentUserMessageIdRef.current,
             }));
           }
           break;
        case 'error': {
           // Backend sends rejection errors at the TOP level ({type:'error', message:...}),
           // older frames used data.payload. Accept both shapes.
           const errMsg = data.message || data.payload?.message || data.payload?.error || data.payload || '';
           const retryable = data.payload?.retryable !== false;
           console.error('Server Error:', errMsg);
           const errText = normalizeConnectionError(
             errMsg,
             t('ws_error_generic', '连接异常，请稍后重试')
           );
           // An explicit server-side error before any usable session means the
           // connection is being rejected (e.g. Invalid scenario after a goal
           // switch). Surface a readable reason instead of an endless
           // "connecting" spinner, and stop auto-reconnect.
           const isRejection = !retryable || /invalid scenario/i.test(errText);
           wsRejectedRef.current = isRejection;
           setWsRejected(isRejection);
           if (isRejection) {
             setIsManualDisconnect(true); // prevent the close handler from auto-reconnecting
             isManualDisconnectRef.current = true;
             if (reconnectTimerRef.current) { clearTimeout(reconnectTimerRef.current); reconnectTimerRef.current = null; }
           }
           setWebSocketError(
             isRejection
               ? t('ws_error_invalid_scenario', '场景无效，请返回重新选择场景')
               : (errText || t('ws_error_rejected', '无法开始本次对话，请返回重新选择场景'))
           );
           break;
        }
        case 'user_proficiency_feedback':
           // Handle proficiency feedback from workflow service
           console.log('📊 Proficiency Feedback:', data.payload);
           break;
        case 'proficiency_update':
           // Handle proficiency update notification with deduplication
           const profPayload = data.payload || {};
           if (!acceptScoringMessage(profPayload)) break;
           const expectedGeneration = scoringGenerationByTaskRef.current.get(String(profPayload.task_id));
           const payloadGeneration = Number(profPayload.scoring_generation);
           const staleGeneration = expectedGeneration !== undefined && (
             !Number.isFinite(payloadGeneration) || payloadGeneration !== expectedGeneration
           );
           if (staleGeneration || !isCompletedWindowEvaluation(profPayload)) {
               console.log('Ignoring non-completed or stale scoring window:', profPayload.evaluation_status);
               break;
           }
           const updateKey = profPayload.evaluation_id || profPayload.turn_id || `${profPayload.task_id}-${profPayload.score ?? profPayload.task_score}-${profPayload.delta}`;

           // Skip if we've already processed this exact update
           if (lastProficiencyUpdateRef.current === updateKey) {
               console.log('Skipping duplicate proficiency update:', updateKey);
               break;
           }
           lastProficiencyUpdateRef.current = updateKey;
           feedbackOrderRef.current.set(`${profPayload.task_id}:${profPayload.scoring_generation ?? 0}`, Number(profPayload.interaction_count || 0));
           setProgressFeedback(profPayload.task_completed ? null : {
             ...profPayload,
             reason: profPayload.reason || profPayload.message || '',
           });

           // Detect task switch and reset progress tracking
           const newTaskId = profPayload.task_id;
           if (newTaskId && lastSeenTaskIdRef.current !== null && newTaskId !== lastSeenTaskIdRef.current) {
               // Only reset if we've already recorded a task_id and it's different (true task switch)
               console.log(`🔄 Task switched: ${lastSeenTaskIdRef.current} → ${newTaskId}, resetting progress`);
               previousProgressRef.current = 0;
               setCurrentTaskProgress(0);
               setCurrentTaskScore(0);
           }
           // Always update ref on first encounter or task switch
           if (newTaskId) {
               lastSeenTaskIdRef.current = newTaskId;
           }

           console.log('📈 Proficiency Update:', profPayload);
           const delta = profPayload.delta || profPayload.proficiency_delta || 0;
           const total = profPayload.total || profPayload.current_proficiency || 0;
           const taskScore = Number(profPayload.score ?? profPayload.task_score ?? 0);
           const message = profPayload.message || '';
           const improvementTips = profPayload.improvement_tips || [];

           // Show improvement tips only outside magic_repetition phase
           if (improvementTips.length > 0 && currentPhaseRef.current !== 'magic_repetition') {
               const tipsText = '💡 ' + improvementTips.join('；');
               setMessages(prev => [...prev, {
                   type: 'system',
                   content: tipsText,
                   isFinal: true,
                   className: 'text-sm text-slate-500'
               }]);
           }

           const taskCompleted = Boolean(profPayload.task_completed);
           if (!profPayload.task_ready_to_complete) {
             setTaskReadyToComplete(current => (
               current && String(current.task_id) === String(profPayload.task_id)
                 ? null
                 : current
             ));
             setTaskCompletionPending(false);
           }
           const newProgress = calculateTaskProgress({
             score: taskScore,
             completedWindowCount: profPayload.completed_window_count,
             taskCompleted,
             previousProgress: previousProgressRef.current,
           });
           setCurrentTaskProgress(newProgress);
           previousProgressRef.current = newProgress;
           setCurrentTaskScore(taskScore);

           // Zero-point windows still record evidence server-side, but do not
           // show a misleading positive-score toast.
           if (delta > 0) {
               const total = profPayload.total || profPayload.current_proficiency || 0;
               const taskScore = Number(profPayload.score ?? profPayload.task_score ?? 0);
               const message = profPayload.message || '';

               // Build message with improvement tips if available
               let content = `+${delta} 熟练度 | 总分：${total}`;
               if (message && message !== '+1 熟练度 | 表现良好，继续保持' && message !== '+2 熟练度 | 表现优秀！继续加油！') {
                   content += ` - ${message}`;
               }

               setMessages(prev => [...prev, {
                   type: 'system',
                   content: content,
                   isFinal: true
               }]);

               // Update engagement level based on delta
               if (delta >= 3) {
                   setEngagementLevel('高');
               } else if (delta >= 2) {
                   setEngagementLevel('中');
               } else {
                   setEngagementLevel('低');
               }
               
               // Save to localStorage for persistence
               const searchParams = new URLSearchParams(window.location.search);
               const scenario = searchParams.get('scenario') || location.state?.scenario;
               if (scenario) {
                   localStorage.setItem(_lsScenarioKey('task_progress_', scenario), newProgress.toString());
               }
               
               // Auto-dismiss after 3 seconds
               setTimeout(() => {
                   setMessages(prev => prev.filter(m => m.type !== 'system' || !m.content.includes('熟练度')));
               }, 3000);
           }

           // Show ScorePopup when a sub-task is completed with scores
           if (profPayload.task_completed && profPayload.scores) {
               setBatchScores(profPayload.scores);
               setLatestDelta(delta);
               setShowScorePopup(true);
           }
           break;
        case 'task_completed':
           // Handle task completion notification
           console.log('✅ Task Completed:', data.payload);
           const taskPayload = data.payload || {};
           if (taskPayload.task_id && !acceptScoringMessage(taskPayload)) break;
           setProgressFeedback(null);
           setTaskCompletionPending(false);
           setTaskReadyToComplete(null);
           if (taskPayload.task_title) {
               // 构建提示消息，包含下一个任务预告
               let completionMessage = `✅ 任务完成：${taskPayload.task_title}`;
               if (taskPayload.next_task) {
                   completionMessage += ` | 下个任务：${taskPayload.next_task}`;
               }
               
               setMessages(prev => [...prev, {
                   type: 'system',
                   content: completionMessage,
                   isFinal: true
               }]);
               // Update completed tasks
               if (taskPayload.task_title) {
                   setCompletedTasks(prev => new Set([...prev, taskPayload.task_title]));
               }
               
               lastProficiencyUpdateRef.current = null; // Reset deduplication for next task

               // next_task is null/undefined when this was the last task in the scenario.
               // In that case keep progress at 100% until the completion modal appears.
               const isLastTask = !taskPayload.next_task;
               const searchParams = new URLSearchParams(window.location.search);
               const scenario = searchParams.get('scenario') || location.state?.scenario;
               if (isLastTask) {
                   setCurrentTaskProgress(100);
                   previousProgressRef.current = 100;
               } else {
                   // Reset progress bar for the next task
                   setCurrentTaskProgress(0);
                   setCurrentTaskScore(0);
                   setEngagementLevel('中');
                   previousProgressRef.current = 0;

                   // Clear localStorage for this scenario (will be refreshed from backend)
                   if (scenario) {
                       localStorage.removeItem(_lsScenarioKey('task_progress_', scenario));
                   }
               }
               
               // Refresh tasks from backend to get next task
               setTimeout(async () => {
                   try {
                       const res = await userAPI.getActiveGoal();
                       if (res && res.goal && res.goal.scenarios) {
                           let activeScenario = res.goal.scenarios.find(s => s.title.trim() === scenario?.trim());
                           
                           // Try case-insensitive match if exact match fails
                           if (!activeScenario) {
                               activeScenario = res.goal.scenarios.find(s => 
                                   s.title.toLowerCase() === scenario?.toLowerCase()
                               );
                           }
                           
                           // Try partial match as fallback
                           if (!activeScenario) {
                               activeScenario = res.goal.scenarios.find(s =>
                                   s.title.toLowerCase().includes(scenario?.toLowerCase() || '') ||
                                   (scenario && scenario.toLowerCase().includes(s.title.toLowerCase()))
                               );
                           }
                           
                           if (activeScenario && activeScenario.tasks) {
                               setTasks(activeScenario.tasks);
                               // Update completed set
                               const newCompleted = new Set();
                               activeScenario.tasks.forEach(t => {
                                   if (typeof t === 'object' && t.status === 'completed') {
                                       newCompleted.add(t.text);
                                   }
                               });
                               setCompletedTasks(newCompleted);
                           }
                       }
                   } catch (err) {
                       console.error('Failed to refresh tasks after completion:', err);
                   }
               }, 1500);
           }
           break;
        case 'phase_transition': {
           const phase = data.payload?.phase || data.phase;
           // recall 模式：魔法重复全部完成后跳回 Dashboard
           if (isRecallMode && phase === 'scene_theater') {
               const today = new Date().toISOString().slice(0, 10);
               localStorage.setItem(`recall_completed_${today}`, 'true');
               navigate('/discovery');
               return;
           }
           if (phase) {
               setCurrentPhase(phase);
               setTipIndex(Math.floor(Math.random() * MAGIC_TIPS.length));
           }
           if (phase === 'magic_repetition') {
               setMagicCardState('waiting');
               setMagicCardCovered(false);
               // stop_audio=false 表示 AI 已在同一响应中合并输出 A+B，不打断正在播放的音频
               // stop_audio=true（或未设置）表示后端另发了 response.create，需要清空旧音频
               if (data.payload?.stop_audio !== false) {
                   stopAudioPlayback();
               }
               // 若 AI 在同一响应中嵌入了 [MAGIC_SENTENCE]，直接从 payload 取句子更新台词卡
               if (data.payload?.magic_sentence) {
                   const sc = new URLSearchParams(window.location.search).get('scenario') || '';
                   setCurrentMagicSentence(data.payload.magic_sentence);
                   try { if (sc) localStorage.setItem(_lsScenarioKey('magic_sentence_', sc), data.payload.magic_sentence); } catch {}
                   console.log('[Magic] Embedded sentence from phase_transition:', data.payload.magic_sentence);
               }
           } else {
               // 切换到其他阶段（scene_theater 等）时重置卡片状态
               setMagicCardCovered(false);
               setMagicCardState('waiting');
           }
           setShowSkipButton(false);
           console.log('📊 Phase Transition:', phase);
           break;
        }
        case 'scene_image': {
           const imageUrl = data.payload?.image_url || data.image_url;
           if (imageUrl) setSceneImageUrl(imageUrl);
           console.log('🖼️ Scene Image:', imageUrl);
           break;
        }
        case 'magic_sentence_update': {
           const newSentence = data.payload?.sentence;
           if (newSentence) {
               const sc = new URLSearchParams(window.location.search).get('scenario') || '';
               setCurrentMagicSentence(newSentence);
               try { if (sc) localStorage.setItem(_lsScenarioKey('magic_sentence_', sc), newSentence); } catch {}
               console.log('[Magic] Sentence updated from AI text:', newSentence);
           }
           break;
        }
        case 'magic_pass_first': {
           setMagicCardCovered(true);
           setMagicCardState('reciting');
           setShowSkipButton(false);
           setTipIndex(Math.floor(Math.random() * MAGIC_TIPS.length));
           console.log('🎭 Magic Pass First — card covered, memory mode');
           break;
        }
        case 'magic_pass': {
           const magicTaskIndex = data.payload?.task_index ?? data.task_index;
           setMagicPassedTasks(prev => {
               const next = new Set([...prev, magicTaskIndex]);
               try {
                   const sc = new URLSearchParams(window.location.search).get('scenario') || '';
                   if (sc) localStorage.setItem(_lsScenarioKey('magic_passed_', sc), JSON.stringify([...next]));
               } catch {}
               return next;
           });
           setMagicCardState('passed');
           setMagicCardCovered(false);
           setShowSkipButton(false);
           setTipIndex(Math.floor(Math.random() * MAGIC_TIPS.length));
           setTimeout(() => {
               setMagicCardState('waiting');
           }, 1800);
           // Stop response A audio and delete its bubble — it may say "try from memory" which
           // conflicts with the new task intro (Response B). Only Response B should appear.
           stopAudioPlayback();
           setMessages(prev => {
               // Find last AI message index via reverse scan (O(n) single pass, no intermediate arrays)
               let lastAiIdx = -1;
               for (let i = prev.length - 1; i >= 0; i--) {
                   if (prev[i].type === 'ai') { lastAiIdx = i; break; }
               }
               if (lastAiIdx === -1) return prev;
               console.log('[magic_pass] Removing Response A bubble at index', lastAiIdx);
               return prev.filter((_, i) => i !== lastAiIdx);
           });
           console.log('✨ Magic Pass Task:', magicTaskIndex);
           break;
        }
        case 'theater_task_complete': {
           const theaterTaskIndex = data.payload?.task_index ?? data.task_index;
           setTheaterCompletedTasks(prev => new Set([...prev, theaterTaskIndex]));
           console.log('🎭 Theater Task Complete:', theaterTaskIndex);
           break;
        }
        case 'scenario_review':
           // Handle scenario review data from backend (when all tasks in scenario are completed)
           console.log('📚 [Scenario Review] 场景完成，获取 AI 点评：', data.payload);
           if (data.payload) {
               // Store review data for personalized AI feedback in completion modal
               setScenarioReviewData(data.payload);
               console.log('📚 AI 点评数据已存储，场景完成时将显示个性化点评');

               // Trigger ScorePopup with consolidated scenario scores (not batch_eval)
               if (data.payload.scores) {
                   setBatchScores(data.payload.scores);
                   setLatestDelta(data.payload.delta || 0);
                   setShowScorePopup(true);
               }
           }
           break;
        case 'test_scenario_review':
           // Handle test scenario review data (for debugging)
           console.log('🧪 [Test Scenario Review] 通关口令生效！');
           console.log('🧪 Test Scenario Review:', data.payload);
           if (data.payload) {
               // Store review data for personalized AI feedback in completion modal
               setScenarioReviewData(data.payload);
               console.log('🧪 测试数据已存储，场景完成时将显示个性化 AI 点评');
           }
           break;
        case 'language_gate_warning': {
           const payload = data.payload || {};
           const msg = payload.message || `请用 ${payload.target_language || '目标语言'} 回答。`;
           console.warn('[DAILY_QA] language gate warning:', payload);
           setLanguageGateWarning({ message: msg, target: payload.target_language || '' });
           break;
        }
        case 'daily_qa_completed':
           console.log('✅ Daily QA Completed', data.payload?.is_bonus ? '(bonus)' : '');
           setDailyQAIsBonus(!!data.payload?.is_bonus);
           setTimeout(() => setShowDailyQAPassModal(true), 800);
           break;
        case 'scoring_feedback':
           if (acceptScoringMessage(data.payload)) {
               if (data.payload.interaction_count != null) feedbackOrderRef.current.set(`${data.payload.task_id}:${data.payload.scoring_generation ?? 0}`, Number(data.payload.interaction_count));
               setProgressFeedback(data.payload);
               setTaskReadyToComplete(null);
               setTaskCompletionPending(false);
           }
           break;
        case 'task_ready_to_complete':
           console.log('🏁 Task Ready to Complete:', data.payload);
           if (acceptScoringMessage(data.payload)) {
               if (data.payload.interaction_count != null) feedbackOrderRef.current.set(`${data.payload.task_id}:${data.payload.scoring_generation ?? 0}`, Number(data.payload.interaction_count));
               setProgressFeedback(null);
               setCompletionSheetDismissed(false);
               setTaskCompletionPending(false);
               setTaskReadyToComplete(data.payload);
           }
           break;
        case 'task_switch_error':
           setTaskCompletionPending(false);
           setMessages(prev => [...prev, {
             type: 'system',
             content: data.payload?.message || '任务切换失败，请重试',
             isFinal: true,
           }]);
           break;
        case 'response.audio.done':
           // Backend signals the current AI turn's TTS is fully delivered.
           // The user is no longer waiting on the model — clear the thinking
           // flag now so the mascot can't get stuck on the thinking face after
           // a turn ends (e.g. welcome / system-continuation turns that never
           // produced a user-driven `isWaitingForAIResponse=true` clear path,
           // or audio that finished without flipping the flag back).
           setIsWaitingForAIResponse(false);
           streamAudioDoneRef.current = (
             data.payload?.responseId
             || data.responseId
             || activeAudioResponseIdRef.current
             || true
           );
           if (
             aiTextReadyForAudioRef.current
             && (
               streamAudioDoneRef.current === true
               || streamAudioDoneRef.current === activeAudioResponseIdRef.current
             )
           ) {
             const scheduler = pcmSchedulerRef.current;
             scheduler?.flush(scheduler.generation).then(() => {
               nextStartTimeRef.current = scheduler.nextStartTime;
             });
           }
           // Schedule the speaking flag to flip off shortly after the last
           // queued chunk finishes — this drives CC subtitle auto-clear.
           if (!receivedStreamAudioRef.current) {
             const ctx = audioContextRef.current;
             const tailMs = ctx
               ? Math.max(0, (nextStartTimeRef.current - ctx.currentTime) * 1000)
               : 0;
             setTimeout(() => {
               const ctxNow = audioContextRef.current?.currentTime ?? 0;
               if (audioQueueRef.current.length === 0 && nextStartTimeRef.current <= ctxNow + 0.05) {
                 setIsAISpeaking(false);
               }
             }, tailMs + 100);
           }
           break;
        case 'dashscope_response':
           // Internal DashScope events - ignore
           break;
        case 'daily_limit_reached': {
           console.log('⛔ Daily limit reached', data);
           const modal = resolveDailyLimitModal(data);
           setDailyLimitModal(modal);
           // 停录，禁止继续发：RealTimeRecorder imperative handle
           try { recorderRef.current?.stopRecording?.(); } catch (_) {}
           break;
        }
        default:
           // Ignore unknown message types silently
           break;
      }
  }, [setCurrentTaskProgress, setCurrentTaskScore, setEngagementLevel, setCompletedTasks, setTasks, location.state, userAPI, acceptScoringMessage]);

  const playAudioChunk = useCallback(async (audioDataOrPromise) => {
    if (isInterruptedRef.current) return; // Drop audio if interrupted
    const contextReady = initAudioContext();
    const scheduler = pcmSchedulerRef.current;
    if (!scheduler) return;
    const generation = scheduler.generation;
    try {
      const accepted = await scheduler.enqueue(
        Promise.resolve(contextReady).then(() => audioDataOrPromise),
        generation
      );
      if (accepted && generation === scheduler.generation) {
        nextStartTimeRef.current = scheduler.nextStartTime;
        speechTotalDurationRef.current = Math.max(
          speechTotalDurationRef.current,
          scheduler.nextStartTime - speechStartTimeRef.current
        );
      }
    } catch (error) {
      // Keep streamedAudioSinceCut=false so the complete COS recording remains
      // eligible as a fallback when PCM conversion/scheduling fails.
      streamedAudioSinceCutRef.current = false;
      console.error('Failed to schedule PCM chunk:', error);
    }
  }, []);

  playAudioChunkRef.current = playAudioChunk;

  // --- WebSocket Logic ---
  const connectWebSocket = useCallback(async (explicitSessionId = null, options = {}) => {
    const { signal, suppressWelcome = false } = options;
    const effectiveSessionId = explicitSessionId || sessionId;
    // Cookie-based auth: check user instead of token
    if (!user || !effectiveSessionId) {
      console.log('connectWebSocket: missing user or sessionId', { user, effectiveSessionId, sessionId });
      return;
    }

    // Store in ref for later use
    connectWebSocketRef.current = connectWebSocket;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    // Destroy the previous wrapper before replacing it. This removes its close
    // listener, heartbeat and timeout callbacks so it cannot schedule a second
    // reconnect after the new socket starts.
    if (socketRef.current) {
        socketRef.current.removeAllListeners();
        socketRef.current.destroy();
        socketRef.current = null;
    }

    // Initialize network adaptive manager
    if (!window.networkAdaptiveManager) {
      window.networkAdaptiveManager = new NetworkAdaptiveManager({
        enableLogging: true,
        onNetworkChange: (networkState) => {
          console.log('Network conditions changed:', networkState);
          // Update UI or streaming quality based on network conditions
        },
        onQualityChange: (newQuality, oldQuality) => {
          console.log('Network quality changed:', { old: oldQuality, new: newQuality });
          // Adapt streaming quality based on network quality
        }
      });
    }

    // Determine WebSocket URL based on environment
    let wsUrl;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const searchParams = new URLSearchParams(window.location.search);
    const scenario = searchParams.get('scenario');
    const voice = localStorage.getItem('ai_voice') || 'Tina';
    const persona = getPersona(voice);
    
    // Determine the correct WebSocket host
    let wsHost;
    if (window.location.hostname === 'localhost' && window.location.port === '3000') {
      // Development server - connect to API gateway on port 8081
      wsHost = 'localhost:8081';
    } else {
      // Production or Docker environment - use current host
      wsHost = window.location.host;
    }
    
    const mode = searchParams.get('mode');
    let realtime;
    try {
      realtime = await conversationAPI.createRealtimeTicket({ signal });
    } catch (error) {
      if (error.name === 'AbortError' || signal?.aborted) return;
      console.error('Failed to create realtime ticket:', error);
      setWebSocketError('无法建立安全连接，请稍后重试');
      return;
    }
    if (signal?.aborted) return;
    wsUrl = `${protocol}//${wsHost}/api/v1/realtime?ticket=${encodeURIComponent(realtime.ticket)}&sessionId=${encodeURIComponent(effectiveSessionId)}${scenario ? `&scenario=${encodeURIComponent(scenario)}` : ''}&voice=${encodeURIComponent(voice)}${mode ? `&mode=${encodeURIComponent(mode)}` : ''}`;

    // Create optimized WebSocket connection
    socketRef.current = new OptimizedWebSocket(wsUrl, {
      reconnectInterval: 1000,
      maxReconnectAttempts: 5,
      connectionTimeout: 30000, // Increased to 30 seconds for AI service connection
      heartbeatInterval: 15000, // Reduced to 15 seconds for more frequent heartbeat
      enableLogging: true,
      enableCompression: true
    });

    // Set up network adaptive manager with WebSocket
    if (window.networkAdaptiveManager) {
      window.networkAdaptiveManager.setWebSocket(socketRef.current);
    }

    // Register event listeners BEFORE connecting to avoid missing events
    socketRef.current.addEventListener('open', () => {
    console.log('WS Open (Optimized)');
    setReconnectAttempts(0);
    reconnectAttemptsRef.current = 0;
    setIsConnected(!isRestoringSessionRef.current);
    setWebSocketError(null);
    // A successful open clears any prior rejection state.
    wsRejectedRef.current = false;
    setWsRejected(false);

    // Note: Ping/heartbeat is handled by OptimizedWebSocket internally
    // No need for manual ping interval here

    // Send session_start handshake
    const searchParams = new URLSearchParams(window.location.search);
    const scenario = searchParams.get('scenario');

    // Check if welcome message should be muted (after retry)
    const welcomeMuted = scenario ? localStorage.getItem(_lsScenarioKey('welcome_muted_', scenario)) === 'true' : false;

      const payload = {
        type: 'session_start',
        userId: user.id,
          sessionId: effectiveSessionId,
        token: token,
        scenario: scenario,
        topic: searchParams.get('topic'),
        mode: searchParams.get('mode'),
          isRestoration: true,
        welcomeMuted: welcomeMuted || suppressWelcome,
        clientInfo: {
            optimized: true,
          version: '2.0',
      features: ['adaptive_streaming', 'compression', 'low_latency']
    }
    };
    socketRef.current.send(JSON.stringify(payload));

    // 刷新重连：如果处于 magic_repetition 且句子为空，请求后端重发
    setTimeout(() => {
      try {
        const sc = new URLSearchParams(window.location.search).get('scenario') || '';
        const hasSentence = sc ? !!localStorage.getItem(_lsScenarioKey('magic_sentence_', sc)) : false;
        if (currentPhaseRef.current === 'magic_repetition' && !hasSentence) {
          socketRef.current.send(JSON.stringify({ type: 'resend_magic_sentence' }));
          console.log('[Magic] Requested resend_magic_sentence after reconnect');
        }
      } catch {}
    }, 600);
    });

    socketRef.current.addEventListener('message', async (event) => {
      console.log('[WS Message] Type:', event.data?.constructor?.name, 'Size:', event.data?.byteLength || event.data?.size || 'N/A');
      
      if (event.data instanceof ArrayBuffer) {
        // Handle binary audio data
        console.log('[Audio] Received binary audio data, size:', event.data.byteLength);
        receivedStreamAudioRef.current = true;
        const packet = unpackPcmAudioPacket(event.data);
        if (
          aiTextReadyForAudioRef.current
          && (!packet.responseId || packet.responseId === activeAudioResponseIdRef.current)
        ) playAudioChunk(packet.pcm);
        else pendingStreamAudioRef.current.push(packet);
      } else if (typeof event.data === 'string') {
        try {
          const data = JSON.parse(event.data);
          handleJsonMessage(data);
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      } else if (event.data instanceof Blob) {
        // Handle blob data
        console.log('[Audio] Received blob data, size:', event.data.size);
        receivedStreamAudioRef.current = true;
        const conversion = event.data.arrayBuffer().then(unpackPcmAudioPacket);
        if (aiTextReadyForAudioRef.current) {
          playAudioChunk(conversion.then(packet => (
            !packet.responseId || packet.responseId === activeAudioResponseIdRef.current
              ? packet.pcm
              : new Uint8Array()
          )));
        } else {
          // Queue the conversion promise immediately so the scheduler preserves
          // WebSocket invocation order even if Blob conversions resolve out of order.
          pendingStreamAudioRef.current.push({ packetPromise: conversion });
        }
      } else {
        console.warn('[WS] Unknown message type:', typeof event.data, event.data);
      }
    });

    socketRef.current.addEventListener('error', (error) => {
        console.error('WebSocket Error (Optimized):', error);
        setWebSocketError('连接异常');
        setIsConnected(false);
    });

    socketRef.current.addEventListener('close', async (event) => {
        console.log('WebSocket Closed (Optimized):', event.code, event.reason);
        setIsConnected(false);

        // Save conversation history when connection closes
        void saveConversationHistory(null, null, { keepalive: true });

        // Stop network monitoring
        if (window.networkAdaptiveManager) {
          window.networkAdaptiveManager.stopMonitoring();
        }

        // Backend rejects an invalid connection (e.g. bad scenario after a goal
        // switch) by closing with a policy-violation code (1008) or an app code
        // (4400). Treat these as "rejected": surface the error, do NOT silently
        // swallow as a clean close, and do NOT auto-reconnect into the same loop.
        const isRejectedClose = event.code === 1008 || event.code === 4400;

        // Don't auto-reconnect if:
        // 1. It was a clean close (code 1000)
        // 2. User manually disconnected
        // 3. Max reconnect attempts reached
        // 4. The connection was rejected by the backend
        const isCleanClose = event.code === 1000 || event.code === 1001;

        const manualDisconnect = isManualDisconnectRef.current;
        const attempts = reconnectAttemptsRef.current;
        if (isCleanClose || isRejectedClose || manualDisconnect || wsRejectedRef.current || attempts >= MAX_RECONNECT_ATTEMPTS) {
            if (isRejectedClose || wsRejectedRef.current) {
                // A rejection. The 'error' message frame usually arrives BEFORE this
                // close and has already set webSocketError — only fill it in if it's
                // still empty so we never overwrite the more specific server reason.
                wsRejectedRef.current = true;
                setWsRejected(true);
                setWebSocketError(prev => prev || t('ws_error_rejected', '无法开始本次对话，请返回重新选择场景'));
            } else if (!isCleanClose && !manualDisconnect && attempts < MAX_RECONNECT_ATTEMPTS) {
                setWebSocketError(`连接已关闭 (${event.code})`);
            } else if (attempts >= MAX_RECONNECT_ATTEMPTS) {
                setWebSocketError(`已达到最大重试次数 (${MAX_RECONNECT_ATTEMPTS})，请刷新页面或点击重试`);
            }
            return;
        }

        // Auto-reconnect with exponential backoff for unexpected disconnections
        if (!manualDisconnect && attempts < MAX_RECONNECT_ATTEMPTS) {
            if (reconnectTimerRef.current) return;
            const attemptNum = attempts + 1;
            console.log(`Attempting automatic reconnection ${attemptNum}/${MAX_RECONNECT_ATTEMPTS}...`);
            setReconnectAttempts(attemptNum);
            reconnectAttemptsRef.current = attemptNum;

            // Exponential backoff: 1s, 2s, 4s, 8s, 10s
            const delay = Math.min(1000 * Math.pow(2, attempts), 10000);

            reconnectTimerRef.current = setTimeout(() => {
                reconnectTimerRef.current = null;
                // Check if we should still reconnect — read the ref .current (not the
                // stale closure snapshot) so a manual retry during the backoff window
                // cancels this pending reconnect instead of firing a spurious one.
                if (!isManualDisconnectRef.current && reconnectAttemptsRef.current <= MAX_RECONNECT_ATTEMPTS) {
                    connectWebSocketRef.current?.(effectiveSessionId);
                }
            }, delay);
        }
    });

    socketRef.current.addEventListener('pong', (data) => {
      console.log('Pong received:', data);
      // Update network metrics
      if (window.networkAdaptiveManager) {
        window.networkAdaptiveManager.handlePong(data);
      }
    });

    socketRef.current.addEventListener('reconnect', (data) => {
      console.log('WebSocket reconnecting:', data);
      setWebSocketError('重新连接中...');
    });

    // Start the connection AFTER all event listeners are registered
    socketRef.current.connect().catch(err => {
      console.error('WebSocket connection failed:', err);
      setWebSocketError('连接失败，请刷新页面重试');
    });

    // Start network monitoring
    if (window.networkAdaptiveManager) {
      window.networkAdaptiveManager.startMonitoring();
    }

  }, [user, sessionId, playAudioChunk, handleJsonMessage]);

  // Init Session
  useEffect(() => {
    // Create AbortController for this effect
    const abortController = new AbortController();

    const init = async () => {
      if (!user?.id) return; // Cookie-based auth: only need user, token is in httpOnly cookie

      // Onboarding Tour demo: render the static UI (incl. mic) but never open a
      // WebSocket or fetch tasks — the tour only highlights the control.
      if (isTourMode) {
        console.log('[Tour] demo mode — skipping WebSocket/AI init');
        return;
      }

      // Don't auto-reconnect on every render - only on initial mount or manual retry
      if (isManualDisconnect) {
        console.log('Manual disconnect detected, skipping auto-init');
        return;
      }

      // Check URL for sessionId (e.g., ?sessionId=...)
      const searchParams = new URLSearchParams(window.location.search);
      const urlSessionId = searchParams.get('sessionId') || searchParams.get('session'); // Support both
      const scenario = searchParams.get('scenario') || location.state?.scenario;
      const topic = searchParams.get('topic');
      let activeGoalId = location.state?.goalId || null;

      // Load daily QA question if in daily_qa mode
      if (isDailyQAMode) {
        try {
          const qaRes = await aiAPI.getDailyQuestion({ signal: abortController.signal });
          if (qaRes && qaRes.question_text) {
            setDailyQAQuestion(qaRes.question_text);
            setDailyQAReferenceAnswer(qaRes.reference_answer || '');
            setDailyQAError(false);
          } else {
            setDailyQAError(true);
          }
        } catch (err) {
          console.error('Failed to load daily QA question:', err);
          setDailyQAError(true);
        }
      }

      // Always refresh tasks from backend to ensure consistency
      if (scenario) {
          try {
              console.log('Fetching tasks from backend for scenario:', scenario);

              const goalRes = await userAPI.getActiveGoal({ signal: abortController.signal });
              console.log('getActiveGoal response:', goalRes);

              let scenarios = [];
              let activeScenario = null;

              if (goalRes?.goal?.id) activeGoalId = goalRes.goal.id;
              if (goalRes && goalRes.goal && goalRes.goal.scenarios) {
                  console.log('Available Scenarios:', goalRes.goal.scenarios.map(s => s.title));
                  console.log('Requested Scenario:', scenario);
                  scenarios = goalRes.goal.scenarios;
                  
                  // Try exact match first
                  activeScenario = goalRes.goal.scenarios.find(s => s.title.trim() === scenario.trim());
                  
                  // Try case-insensitive match
                  if (!activeScenario) {
                      activeScenario = goalRes.goal.scenarios.find(s => 
                          s.title.toLowerCase() === scenario.toLowerCase()
                      );
                  }
                  
                  // Try partial match as fallback
                  if (!activeScenario) {
                      activeScenario = goalRes.goal.scenarios.find(s =>
                          s.title.toLowerCase().includes(scenario.toLowerCase()) ||
                          scenario.toLowerCase().includes(s.title.toLowerCase())
                      );
                  }
                  
                  console.log('Found active scenario:', activeScenario);
              } else {
                  console.log('No goal found, using default scenarios');

                  // Determine goal type from scenario name
                  let goalType = 'daily_conversation'; // default
                  if (scenario.toLowerCase().includes('business') || scenario.toLowerCase().includes('meeting')) {
                      goalType = 'business_meeting';
                  } else if (scenario.toLowerCase().includes('travel') || scenario.toLowerCase().includes('airport')) {
                      goalType = 'travel_survival';
                  } else if (scenario.toLowerCase().includes('exam') || scenario.toLowerCase().includes('test')) {
                      goalType = 'exam_prep';
                  } else if (scenario.toLowerCase().includes('presentation') || scenario.toLowerCase().includes('speech')) {
                      goalType = 'presentation';
                  }

                  scenarios = DEFAULT_SCENARIOS[goalType] || DEFAULT_SCENARIOS.daily_conversation;
                  activeScenario = scenarios.find(s => s.title.trim() === scenario.trim());

                  if (!activeScenario) {
                      // Try to find a similar scenario
                      activeScenario = scenarios.find(s =>
                          s.title.toLowerCase().includes(scenario.toLowerCase()) ||
                          scenario.toLowerCase().includes(s.title.toLowerCase())
                      );
                  }

                  if (!activeScenario && scenarios.length > 0) {
                      // Use first scenario as fallback
                      activeScenario = scenarios[0];
                  }
              }

              if (activeScenario && activeScenario.tasks) {
                  console.log('Setting tasks from active scenario:', activeScenario.tasks);
                  setTasks(activeScenario.tasks);
                  console.log('Restored tasks from active goal:', activeScenario.tasks);
              } else {
                  console.warn('Scenario not found in active goal or no tasks');
                  console.log('All scenarios:', scenarios);
              }
          } catch (e) {
              console.error('Failed to restore tasks from goal:', e);
              console.error('Error details:', e.message, e.stack);

              // Final fallback - use default scenarios
              console.log('Using default scenarios due to error');
              const defaultScenarios = DEFAULT_SCENARIOS.daily_conversation;
              const activeScenario = defaultScenarios.find(s => s.title.trim() === scenario.trim()) || defaultScenarios[0];

              if (activeScenario && activeScenario.tasks) {
                  setTasks(activeScenario.tasks);
                  console.log('Restored tasks from default scenarios:', activeScenario.tasks);
              }
          }
      }

      // Determine session ID priority: URL > user-scoped localStorage > server-created session.
      // New IDs must come from conversation-service so the matching history
      // document exists before a refresh attempts to restore it.
      const sessionKey = scenario && user?.id ? _lsSessionKey(user.id, scenario) : null;
      const legacySessionKey = scenario ? _lsScenarioKey('session_', scenario) : null;
      const storedSessionId = sessionKey
        ? (localStorage.getItem(sessionKey) || localStorage.getItem(legacySessionKey))
        : null;
      let effectiveSessionId = urlSessionId || storedSessionId;
      let restoredHistory = false;

      // Restore both URL-selected sessions and locally persisted sessions.
      // Previously URL sessions skipped this branch, so opening an existing
      // conversation connected the socket without rendering its messages.
      if (effectiveSessionId) {
              try {
                  const historyRes = await conversationAPI.getHistory(effectiveSessionId, { signal: abortController.signal });
                  if (historyRes?.success && historyRes.messages) {
                      // Load history messages into state
                      // Set audioPlayed: true to prevent auto-play on page refresh
                      let lastMagicSentence = '';
                      const mappedHistoryMessages = historyRes.messages.map(msg => {
                          let content = msg.content || '';
                          if (msg.role !== 'user') {
                              // 提取最新的 MAGIC_SENTENCE（取最后一条，共享正则兼容嵌套括号）
                              const magic = extractMagicSentence(content);
                              if (magic) lastMagicSentence = magic;
                              // 剥离所有标记（共享单一定义点）
                              content = stripAllMarkers(content).trim();
                          }
                          return {
                              type: msg.role === 'user' ? 'user' : 'ai',
                              content,
                              audioUrl: msg.audioUrl,
                              isFinal: true,
                              audioPlayed: true,
                              historyId: msg.id || msg._id,
                              scenario: msg.scenario,
                              task_id: msg.task_id,
                              turn_id: msg.turn_id
                          };
                      });
                      const historyMessages = collapseAdjacentHistoryDuplicates(mappedHistoryMessages);
                      restoredHistory = historyMessages.length > 0;
                      restoredAiContentKeysRef.current = new Set(
                        historyMessages
                          .filter(message => message.type === 'ai')
                          .map(historyContentKey)
                      );
                      if (lastMagicSentence) {
                          setCurrentMagicSentence(lastMagicSentence);
                      }
                      setMessages(prev => {
                          // Keep initial system message, add history
                          const systemMsg = prev.find(m => m.type === 'system');
                          return systemMsg ? [systemMsg, ...historyMessages] : historyMessages;
                      });
                      console.log('Loaded history messages:', historyMessages.length);
                      if (sessionKey) localStorage.setItem(sessionKey, effectiveSessionId);
                      if (legacySessionKey && localStorage.getItem(legacySessionKey) === effectiveSessionId) {
                        localStorage.removeItem(legacySessionKey);
                      }
                  } else if (historyRes?.status === 403 || historyRes?.status === 404) {
                      if (sessionKey) localStorage.removeItem(sessionKey);
                      if (legacySessionKey) localStorage.removeItem(legacySessionKey);
                      effectiveSessionId = null;
                  }
              } catch (err) {
                  console.log('Failed to load history:', err);
                  if (err?.status === 403 || err?.status === 404) {
                      if (sessionKey) localStorage.removeItem(sessionKey);
                      if (legacySessionKey) localStorage.removeItem(legacySessionKey);
                      effectiveSessionId = null;
                  }
              }
      }

      // If still no session ID, create and initialize it on the server. Keep a
      // local UUID only as an availability fallback when session creation is
      // temporarily unreachable; autosave will still persist that fallback.
      if (!effectiveSessionId) {
          restoredHistory = false;
          try {
              const created = await conversationAPI.startSession(
                { goalId: activeGoalId, forceNew: true },
                { signal: abortController.signal }
              );
              if (!created?.sessionId) throw new Error('Session service returned no sessionId');
              effectiveSessionId = created.sessionId;
          } catch (err) {
              if (abortController.signal.aborted) return;
              console.error('Failed to initialize conversation session:', err);
              effectiveSessionId = crypto.randomUUID();
          }
      }

      // Persist session ID for this scenario
      if (sessionKey) localStorage.setItem(sessionKey, effectiveSessionId);

      setSessionId(effectiveSessionId);
      isRestoringSessionRef.current = restoredHistory;
      setIsRestoringSession(restoredHistory);
      
      // Set current scenario info
      if (scenario) {
          setCurrentScenarioTitle(scenario);
      }
      if (location.state?.allScenarios) {
          setAllScenarios(location.state.allScenarios);
      }
      if (location.state?.currentIndex !== undefined) {
          setCurrentScenarioIndex(location.state.currentIndex);
      }

      // Connect WebSocket with the effective session ID (state may not be updated yet)
      connectWebSocket(effectiveSessionId, {
        signal: abortController.signal,
        suppressWelcome: restoredHistory,
      });
    };

    init();
    
    // Cleanup function to prevent memory leaks and stale callbacks
    return () => {
      console.log('[Cleanup] Conversation component unmounting, cleaning up resources...');

      // Abort any pending API requests
      abortController.abort();

      // Close WebSocket connection
      if (socketRef.current) {
        console.log('[Cleanup] Closing WebSocket connection');
        socketRef.current.removeAllListeners();
        socketRef.current.destroy();
        socketRef.current = null;
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      // Stop any ongoing audio playback
      stopAudioPlayback();

      // Stop network monitoring
      if (window.networkAdaptiveManager) {
        window.networkAdaptiveManager.stopMonitoring();
      }

      // Clear any pending audio queue
      if (audioQueueRef.current) {
        audioQueueRef.current = [];
      }

      // Cleanup daily QA navigation timeout
      if (navTimeoutRef.current) {
        clearTimeout(navTimeoutRef.current);
        navTimeoutRef.current = null;
      }
    };
  }, [token, user, isManualDisconnect]); // Removed connectWebSocket from dependencies to prevent infinite loop

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Persist the latest conversation while the page is still alive so a refresh
  // does not depend on the websocket close handler completing in time.
  useEffect(() => {
    if (!sessionId || !user?.id || messages.length === 0) return;

    if (historyAutosaveTimerRef.current) {
      clearTimeout(historyAutosaveTimerRef.current);
    }

    historyAutosaveTimerRef.current = setTimeout(() => {
      void saveConversationHistory(sessionId, messages, { keepalive: true });
    }, 1200);

    return () => {
      if (historyAutosaveTimerRef.current) {
        clearTimeout(historyAutosaveTimerRef.current);
        historyAutosaveTimerRef.current = null;
      }
    };
  }, [sessionId, user?.id, messages]);

  // Auto-dismiss the language gate warning after 6s.
  useEffect(() => {
    if (!languageGateWarning) return;
    const t = setTimeout(() => setLanguageGateWarning(null), 6000);
    return () => clearTimeout(t);
  }, [languageGateWarning]);

  // Stable signal for the auto-play effect below. Only changes when the SET of
  // pending (audioPlayed===false && audioUrl) AI messages changes — appends of
  // proficiency/translation/task system messages that don't add a playable
  // bubble leave this key untouched, so the effect doesn't re-run and re-stop
  // audio that's currently playing (cut-off chain C).
  const pendingAutoPlayKey = React.useMemo(() => {
    const parts = [];
    messages.forEach((m, i) => {
      if (m.type === 'ai' && m.audioUrl && m.audioPlayed === false) {
        parts.push(`${i}:${m.responseId || ''}:${m.audioUrl}`);
      }
    });
    return parts.join('|');
  }, [messages]);

  // Auto-play AI audio when messages get audio URLs
  useEffect(() => {
    messages.forEach((message, index) => {
      // Only auto-play if explicitly marked as not played (audioPlayed === false)
      // Don't auto-play if audioPlayed is undefined or true
      if (message.type === 'ai' && message.audioUrl && message.audioPlayed === false) {
        // Mark message as played to prevent repeated playback
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[index] = { ...newMessages[index], audioPlayed: true };
          return newMessages;
        });

        // Play the high-quality COS URL via the auto-queue path: it schedules
        // after whatever is currently playing (nextStartTimeRef) for a seamless
        // A→B handoff. Do NOT stopAudioPlayback() here — that would wipe the
        // queue (cut-off chains A & B). Double-playback is already prevented by
        // streamedAudioSinceCutRef (streaming dedupe) and audioPlayed.
        console.log(`[AutoPlay] Playing AI audio for message ${index}:`, message.audioUrl);
        isInterruptedRef.current = false;
        playFullAudio(message.audioUrl, true);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingAutoPlayKey]);

  // Handle text selection for TTS
  useEffect(() => {
    document.addEventListener('mouseup', handleTextSelection);
    return () => document.removeEventListener('mouseup', handleTextSelection);
  }, []);

  // --- Recorder Callbacks ---

  const handleRecordingStart = () => {
    // CRITICAL: Check WebSocket connection before allowing recording
    const wsReadyState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
    if (!isConnected || wsReadyState !== WebSocket.OPEN) {
        console.error('❌ Cannot start recording: WebSocket not connected, state:', wsReadyState);
        alert('AI 导师尚未连接，请稍后再试');
        return;
    }
    setIsUserRecording(true);
    // Starting a new turn cancels any pending `thinking` from the previous one.
    setIsWaitingForAIResponse(false);
    if (!practiceStartTimeRef.current) practiceStartTimeRef.current = Date.now();

    isInterruptedRef.current = false; // Reset flag for new turn
    const newId = Date.now().toString();
    currentUserMessageIdRef.current = newId; // New turn ID

    // Get session ID from recorder (generated in startRecording)
    const newSessionId = recorderRef.current?.getSessionId();
    if (!newSessionId) {
        console.error('❌ No session ID available from recorder');
        return;
    }
    currentRecordingSessionIdRef.current = newSessionId;
    console.log('🎤 Recording started, session ID:', newSessionId);

    // Always stop audio playback immediately (interrupt AI response)
    stopAudioPlayback();
    isInterruptedRef.current = true; // Mark as interrupted

    // 1. Force finalize ALL previous messages
    // 2. Immediately create a placeholder for the NEW user turn
    setMessages(prev => {
        const newMessages = prev.map(msg =>
            (!msg.isFinal) ? { ...msg, isFinal: true, isInterrupted: true } : msg
        );
        return [...newMessages, {
            type: 'user',
            content: '...', // Placeholder content
            isFinal: false,
            id: newId,
            audioUrl: null
        }];
    });

    // Send interruption signal to backend
    console.log('🔇 Interruption triggered - stopping AI response and starting new turn');
    if (socketRef.current?.getReadyState?.() === WebSocket.OPEN || socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
  };

  const handleRecordingStop = (audioBuffers) => {
    setIsUserRecording(false);
    // User finished speaking — show `thinking` until AI audio starts playing.
    setIsWaitingForAIResponse(true);
    const wsReadyState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
    console.log('🎤 handleRecordingStop called, WebSocket state:', wsReadyState, 'audio buffers:', audioBuffers?.length);

    // Clear recording session ID to prevent any late audio data from being sent
    console.log('🛑 Recording stopped, clearing session ID:', currentRecordingSessionIdRef.current);
    currentRecordingSessionIdRef.current = null;

    // Clear session ID in recorder
    if (recorderRef.current) {
        recorderRef.current.clearSessionId();
    }

    // Send buffered audio data as JSON-wrapped base64
    if (audioBuffers && audioBuffers.length > 0) {
        console.log('🎤 Sending buffered audio data, count:', audioBuffers.length);
        audioBuffers.forEach(buffer => {
            if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
                const uint8 = new Uint8Array(buffer.buffer !== undefined ? buffer.buffer : buffer);
                let binary = '';
                for (let i = 0; i < uint8.byteLength; i++) {
                    binary += String.fromCharCode(uint8[i]);
                }
                const b64 = btoa(binary);
                socketRef.current.send(JSON.stringify({
                    type: 'audio_stream',
                    payload: { audio: b64 }
                }));
            } else {
                console.warn('⚠️ Cannot send buffered audio - WebSocket not connected, state:', wsReadyState);
            }
        });
    } else {
        console.log('🎤 No buffered audio to send');
    }

    // Wait for WebSocket to be ready before sending
    if (wsReadyState !== WebSocket.OPEN) {
        console.log('⏳ WebSocket not ready (state:', wsReadyState, '), waiting for connection...');

        // Wait for connection with timeout
        const waitForConnection = () => {
            const checkReady = () => {
                const currentState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
                if (currentState === WebSocket.OPEN) {
                    console.log('✅ WebSocket now ready, sending user_audio_ended');
                    // Re-enable streaming playback for the AI reply that follows.
                    // handleRecordingStart set isInterruptedRef=true to interrupt the
                    // previous turn; without this reset every streaming PCM chunk of
                    // the new reply is dropped in playAudioChunk and the user only
                    // hears the (much later) COS audio_url.
                    isInterruptedRef.current = false;
                    socketRef.current.send(JSON.stringify({ type: 'user_audio_ended' }));
                } else if (currentState === WebSocket.CONNECTING) {
                    // Still connecting, check again in 100ms
                    setTimeout(checkReady, 100);
                } else {
                    console.error('❌ WebSocket failed to connect (state:', currentState, ')');
                    setWebSocketError('连接失败，请刷新页面重试');
                }
            };
            checkReady();
        };

        // Set timeout to give up after 10 seconds
        setTimeout(() => {
            const finalState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
            if (finalState !== WebSocket.OPEN) {
                console.error('❌ WebSocket connection timeout after waiting');
                setWebSocketError('连接超时，请刷新页面重试');
            }
        }, 10000);

        waitForConnection();
        return;
    }

    console.log('📤 Sending user_audio_ended');
    // Re-enable streaming playback for the AI reply that follows. See the
    // matching reset in the waitForConnection path above.
    isInterruptedRef.current = false;
    socketRef.current.send(JSON.stringify({ type: 'user_audio_ended' }));
    console.log('✅ user_audio_ended sent, keeping WebSocket open for AI response');
  };

  const handleRecordingCancel = () => {
    setIsUserRecording(false);
    setIsWaitingForAIResponse(false);
    console.log('🚫 Recording cancelled, clearing session ID:', currentRecordingSessionIdRef.current);
    currentRecordingSessionIdRef.current = null; // Clear session ID to ignore any pending audio data

    const wsReadyState = socketRef.current?.getReadyState?.() || socketRef.current?.readyState;
    if (wsReadyState === WebSocket.OPEN || wsReadyState === WebSocket.CONNECTING) {
        socketRef.current.send(JSON.stringify({ type: 'user_audio_cancelled' }));
    }
    const cancelId = currentUserMessageIdRef.current;
    setMessages(prev => prev.filter(m => !(m.type === 'user' && m.id === cancelId)));
    isInterruptedRef.current = false;
  };

  // Removed handleAudioData since we now cache audio and send only on stop

  // PhaseIndicator 组件：两阶段 tab 指示器（Minimalist 风格）
  const PhaseIndicator = () => {
    const phases = [
      { id: 'magic_repetition', label: '魔法重复', count: 3 },
      { id: 'scene_theater', label: '情景剧场', count: 3 },
    ];

    const handlePhaseClick = (phaseId) => {
      if (phaseId === 'magic_repetition' && currentPhase !== 'magic_repetition') {
        // 回退到魔法重复阶段：重置前端状态 + 通知后端
        setCurrentPhase('magic_repetition');
        setMagicPassedTasks(new Set());
        setCurrentMagicSentence('');
        setMagicCardState('waiting');
        setMagicCardCovered(false);
        try {
          const sc = new URLSearchParams(window.location.search).get('scenario') || '';
          if (sc) localStorage.removeItem(_lsScenarioKey('magic_passed_', sc));
          if (sc) localStorage.removeItem(_lsScenarioKey('magic_sentence_', sc));
        } catch {}
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(JSON.stringify({ type: 'reset_magic_phase' }));
          console.log('🔄 Reset magic phase requested');
        }
      } else {
        setCurrentPhase(phaseId);
      }
    };

    return (
      <div className="h-12 flex items-center gap-2 px-4 bg-slate-900 dark:bg-slate-950 justify-center transition-all duration-500">
        {phases.map((phase) => {
          const isActive = currentPhase === phase.id;
          const completedTasks = phase.id === 'magic_repetition' ? magicPassedTasks : theaterCompletedTasks;
          return (
            <button
              key={phase.id}
              onClick={() => handlePhaseClick(phase.id)}
              className={`flex flex-col items-center gap-1 px-4 transition-colors duration-300 ${
                isActive ? 'text-indigo-400' : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              <div className="text-xs font-medium">{phase.label}</div>
              <div className="flex gap-0.5">
                {Array.from({ length: phase.count }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-1 h-1 rounded-full transition-colors duration-300 ${
                      completedTasks.has(i) ? 'bg-emerald-500' : 'bg-slate-600'
                    }`}
                  />
                ))}
              </div>
            </button>
          );
        })}
      </div>
    );
  };

  // AIFeedbackStrip 组件：简洁的反馈条（底部淡入）
  const AIFeedbackStrip = () => {
    if (!aiFeedback) return null;
    return (
      <div className="px-4 py-2 bg-amber-950/30 dark:bg-amber-900/20 border-t border-amber-800/30 text-amber-200 text-xs flex items-center gap-2 animate-in fade-in duration-500">
        {/* 闪烁点动画 */}
        <span className="inline-flex gap-1">
          <span className="w-0.5 h-0.5 rounded-full bg-amber-400 animate-pulse" />
          <span className="w-0.5 h-0.5 rounded-full bg-amber-400 animate-pulse delay-100" />
          <span className="w-0.5 h-0.5 rounded-full bg-amber-400 animate-pulse delay-200" />
        </span>
        <span className="flex-1">{aiFeedback}</span>
      </div>
    );
  };

  // AiAvatar status derived from recording / AI speaking states.
  // `thinking` is shown ONLY while the user is actually waiting on the model
  // (driven by `isWaitingForAIResponse`, set when the user input is sent and
  // cleared once the AI response/audio arrives).
  //
  // Previously we also showed thinking whenever the last AI message had no
  // audioUrl. That fallback got stuck permanently when an AI turn never
  // produced COS audio (marker-only / system-continuation / dropped upload),
  // leaving the mascot frozen on `bird-expression-thinking.svg` after the
  // round ended. Tying thinking to `isWaitingForAIResponse` lets the mascot
  // fall back to idle (`bird-logo.svg`) as soon as playback finishes.
  const avatarStatus = isAISpeaking ? 'speaking'
    : isUserRecording ? 'listening'
    : isWaitingForAIResponse ? 'thinking'
    : 'idle';

  return (
    <div className="flex h-[100dvh] max-h-[100dvh] flex-col overflow-hidden max-w-lg mx-auto bg-background-light dark:bg-background-dark relative">

      {/* ── Header: 场景图（有时）+ 简洁 nav bar ── */}
      <div className="w-full shrink-0">

        {/* 场景图：仅当后端推送了真实图片时显示 */}
        {sceneImageUrl && (
          <div className="w-full overflow-hidden" style={{ height: '180px' }}>
            <img
              src={sceneImageUrl}
              alt="scene"
              className="w-full h-full object-cover"
            />
          </div>
        )}

        {/* Nav bar：白色底，场景名 + 进度点 + AI状态 */}
        <header className="flex items-center justify-between px-4 bg-white border-b border-gray-100 shadow-sm" style={{ height: '56px' }}>

          {/* 左：× 关闭 + 场景名 */}
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <button
              onClick={() => navigate('/discovery')}
              aria-label={t('qa_ui.conversation_back', '返回发现页')}
              className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 hover:bg-gray-200 transition shrink-0">
              <span className="material-symbols-outlined" aria-hidden="true" style={{ fontSize: '16px' }}>close</span>
            </button>
            <h1 className="text-gray-900 font-semibold text-sm leading-tight truncate">
              {isDailyQAMode ? t('qa_ui.conversation_daily_qa') : (currentScenarioTitle || t('qa_ui.conversation_tutor'))}
            </h1>
          </div>

          {/* 右：子任务进度点 + AI 状态 */}
          <div className="flex items-center gap-3 shrink-0">

            {/* 任务完成进度点（recall模式显示复述进度，普通模式显示场景进度；daily_qa 无子任务，不显示） */}
            {isDailyQAMode ? null : isRecallMode ? (
              <div className="flex gap-1">
                {[0,1,2].map(i => (
                  <div key={i} className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                    i <= magicPassedTasks.size ? 'bg-[#637FF1]' : 'bg-gray-200'
                  }`} />
                ))}
              </div>
            ) : (
              <div className="flex gap-1">
                {[0,1,2].map(i => (
                  <div key={i} className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                    i <= theaterCompletedTasks.size ? 'bg-[#637FF1]' : 'bg-gray-200'
                  }`} />
                ))}
              </div>
            )}

            {/* AI 导师状态（Tour demo 态不连 WS，显"演示"而非误导的"连接中"；
                被后端拒绝时显红色"已断开"而非永久"连接中"） */}
            <span role="status" className={`text-xs px-2 py-0.5 rounded-full flex items-center gap-1 ${
              isTourMode ? 'bg-violet-50 text-violet-600'
                : wsRejected ? 'bg-red-50 text-red-600'
                : isRestoringSession ? 'bg-amber-50 text-amber-700'
                : isConnected ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${
                isTourMode ? 'bg-violet-500' : wsRejected ? 'bg-red-500' : isConnected ? 'bg-emerald-500' : 'bg-amber-400'
              }`} />
              {isTourMode ? t('qa_ui.conversation_demo') : wsRejected ? t('ws_status_rejected', '已断开') : isRestoringSession ? t('ws_status_restoring', '正在恢复对话') : isConnected ? t('qa_ui.conversation_online') : t('qa_ui.conversation_connecting')}
            </span>
          </div>
        </header>
      </div>

      {/* 每日鼓励 Banner（软提示，非硬限制；真正的轮次硬护栏由后端 daily_limit_reached 负责）。
          dailyScenariosUsed 计的是场景数（每完成 1 场景的完成弹窗 +1），每场景含 3 个 task。
          原文案硬编码「3 个场景」会被误读为任务数 —— 改成动态显示已完成的 task 数
          （场景数 ×3），文案明确用「任务」。阈值仍为完成 3 场景（=9 任务）的软上限提示。 */}
      {dailyScenariosUsed >= 3 && !isDailyQAMode && (
        <div className="flex items-center justify-center px-4 py-1.5 bg-emerald-900/30 text-emerald-300 text-xs">
          🎉 今天已完成 {dailyScenariosUsed * 3} 个任务（{dailyScenariosUsed} 个场景），状态很棒！想继续可以接着练
        </div>
      )}

      {/* Daily QA Mode: Fixed question card at top */}
      {isDailyQAMode && (
        <div className="px-4 py-3 bg-indigo-50 border-b border-indigo-100 shrink-0">
          {dailyQAError ? (
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm text-gray-700">今日问题加载中...请稍候</p>
              </div>
              <button
                onClick={() => navigate('/discovery')}
                className="text-xs text-indigo-600 hover:text-indigo-700 font-medium">
                {t('qa_ui.conversation_back')}
              </button>
            </div>
          ) : dailyQAQuestion ? (
            <div className="flex items-center gap-2">
              <span className="text-lg flex-shrink-0">🔊</span>
              <div className="flex-1">
                <p className="text-sm text-gray-700 line-clamp-3">{dailyQAQuestion}</p>
                {dailyQAReferenceAnswer && (
                  <div className="mt-2">
                    <button
                      onClick={() => setShowReferenceAnswer(!showReferenceAnswer)}
                      className="text-xs font-medium"
                      style={{ color: '#6366F1', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
                      {showReferenceAnswer ? '隐藏参考答案 ▲' : '查看参考答案 ▼'}
                    </button>
                    {showReferenceAnswer && (
                      <p className="mt-1 text-xs italic rounded-lg px-3 py-2"
                         style={{ color: '#6B7280', background: '#EEF2FF' }}>
                        💡 {dailyQAReferenceAnswer}
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Mission Tasks Dropdown Bar — only visible in scene_theater phase (not in daily_qa mode) */}
      {!isDailyQAMode && currentPhase === 'scene_theater' && (tasks.length > 0 || tasksLoading) && (
        <nav
          aria-label="场景任务进度"
          className={`z-10 bg-[#4055B5] border-b border-indigo-300/30 transition-all duration-700 shrink-0 relative ${!showTasks && taskBarFaded ? 'opacity-50' : 'opacity-100'}`}
          onMouseEnter={() => {
            if (!showTasks && taskBarFaded) {
              clearTimeout(taskBarFadeTimerRef.current);
              setTaskBarFaded(false);
              taskBarFadeTimerRef.current = setTimeout(() => setTaskBarFaded(true), 3000);
            }
          }}
        >
          {/* Collapsed / Expanded header button */}
          <button
            onClick={() => {
              setShowTasks(prev => {
                const next = !prev;
                clearTimeout(taskBarFadeTimerRef.current);
                if (!next) {
                  taskBarFadeTimerRef.current = setTimeout(() => setTaskBarFaded(true), 3000);
                } else {
                  setTaskBarFaded(false);
                }
                return next;
              });
            }}
            aria-expanded={showTasks}
            aria-controls="conversation-task-list"
            className="w-full bg-[#4055B5] hover:bg-[#35489F] active:bg-[#2D44CA] transition-colors"
          >
            <div className="flex items-center justify-between px-5 py-3">
              <div className="flex items-center gap-3">
                <span className="text-base font-bold text-white">
                  任务 ({completedTasks.size}/{tasks.length} 完成)
                </span>
              </div>
              <span aria-hidden="true" className="material-symbols-outlined text-white text-base">
                {showTasks ? 'expand_less' : 'expand_more'}
              </span>
            </div>
            {/* Overall progress bar — always visible in header */}
            <div
              role="progressbar"
              aria-label="当前子任务进度"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(currentTaskProgress)}
              className="w-full h-1 bg-indigo-950/40"
            >
              <div
                className="h-full bg-white/80 transition-all duration-500 ease-out"
                style={{ width: `${currentTaskProgress}%` }}
              />
            </div>
          </button>

          {showTasks && (
            <ul id="conversation-task-list" aria-label="场景子任务" className="bg-[#4055B5] px-5 pb-4 pt-2 space-y-3 border-t border-indigo-300/30">
              {tasksLoading ? (
                <li className="text-xs text-white/80 py-1">{t('conversation_tasks_loading', '任务加载中…')}</li>
              ) : (() => {
                // determine which is the current in-progress task (first incomplete)
                const firstIncompleteIdx = tasks.findIndex(t => {
                  const txt = typeof t === 'string' ? t : t.text;
                  return !completedTasks.has(txt);
                });
                return tasks.map((task, idx) => {
                  const taskText = typeof task === 'string' ? task : task.text;
                  const isCompleted = completedTasks.has(taskText);
                  const isCurrent = idx === firstIncompleteIdx;
                  const progress = isCompleted ? 100 : isCurrent ? currentTaskProgress : 0;
                  return (
                    <li key={idx} className="space-y-1.5">
                      <div className="flex items-start gap-2.5">
                        <span className={`w-3 h-3 rounded-full flex items-center justify-center shrink-0 mt-1 ${isCompleted ? 'bg-white' : 'bg-white/20'}`}>
                          {isCompleted && (
                            <span className="material-symbols-outlined text-[#637FF1] text-[9px] font-bold">check</span>
                          )}
                        </span>
                        <div className="flex-1 min-w-0">
                          <span className={`text-sm ${isCompleted ? 'text-indigo-100 line-through' : 'text-white font-medium'}`}>
                            {isCurrent && !isCompleted && '→ '}
                            {taskText}
                          </span>
                          {/* Per-task progress bar */}
                          <div className="w-full h-1 mt-1 bg-indigo-900/30 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ease-out ${isCompleted ? 'bg-white' : 'bg-white/70'}`}
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                          {isCurrent && !isCompleted && (
                            <span className="text-xs font-medium text-white mt-0.5 inline-block">{progress}%</span>
                          )}
                        </div>
                      </div>
                    </li>
                  );
                });
              })()}
            </ul>
          )}
        </nav>
      )}

      {/* Floating Playback Button */}
      {!ccMode && !isTourMode && !isRecallMode && !isDailyQAMode && currentPhase === 'scene_theater'
        && activeScoringTask && currentTaskScore >= 9 && currentTaskProgress < 100 && (
        <TaskProgressGuidance
          feedback={progressFeedback && acceptScoringMessage(progressFeedback) ? progressFeedback : null}
          taskTitle={activeScoringTask.text}
          ready={Boolean(taskReadyToComplete && acceptScoringMessage(taskReadyToComplete))}
          onConfirm={() => setCompletionSheetDismissed(false)}
        />
      )}
      {selection.visible && (
        <button
          onClick={playSelectedText}
          aria-label={isSynthesizing ? '正在生成选中文本的语音' : '播放选中文本'}
          className="fixed z-50 p-2 bg-primary text-white rounded-full shadow-lg transform -translate-x-1/2 flex items-center justify-center animate-in fade-in zoom-in duration-200"
          style={{ left: selection.x, top: selection.y }}
        >
          {isSynthesizing ? (
            <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
          ) : (
            <span className="material-symbols-outlined text-xl">volume_up</span>
          )}
        </button>
      )}

      {/* Messages Area */}
      <main className="min-h-0 flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth">

        {/* 魔法重复阶段：台词卡（句子跟读卡片，仅 recall 模式显示）*/}
        {isRecallMode && currentPhase === 'magic_repetition' && (
          <div>
            {/* 台词卡主体 */}
            <div
              className={`bg-white rounded-2xl shadow-md transition-all duration-300 overflow-hidden ${
                magicCardState === 'passed' ? 'border-l-4 border-emerald-500' : 'border-l-4 border-primary'
              }`}
              onPointerDown={() => magicCardCovered && setIsPeeking(true)}
              onPointerUp={() => setIsPeeking(false)}
              onPointerLeave={() => setIsPeeking(false)}
              style={{ userSelect: 'none', touchAction: 'none' }}
            >
              <div className="p-5">
                {magicCardState === 'passed' ? (
                  /* 通过状态 */
                  <div className="flex items-center gap-3 py-1 animate-in zoom-in duration-200">
                    <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center shrink-0">
                      <span className="text-xl">✅</span>
                    </div>
                    <div>
                      <p className="font-semibold text-emerald-700">通过！</p>
                      <p className="text-sm text-emerald-600/70">继续下一个句子</p>
                    </div>
                  </div>
                ) : magicCardCovered && !isPeeking ? (
                  /* 背诵模式（台词已隐藏）*/
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center shrink-0">
                      <span className="text-lg">🧠</span>
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900 mb-2">从记忆复述</p>
                      <div className="space-y-2">
                        <div className="h-3 rounded-full bg-indigo-100 animate-pulse w-full" />
                        <div className="h-3 rounded-full bg-indigo-100 animate-pulse w-4/5" />
                      </div>
                      <p className="text-xs text-gray-400 mt-3">💡 按住卡片可偷看</p>
                    </div>
                  </div>
                ) : magicCardCovered && isPeeking ? (
                  /* 偷看模式 */
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center shrink-0">
                      <span className="text-lg">👀</span>
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-indigo-600 mb-1.5 text-sm">偷看模式</p>
                      <p className="text-gray-700 leading-relaxed text-[15px]">
                        {currentMagicSentence || <span className="text-slate-400 italic">AI 正在准备...</span>}
                      </p>
                      <p className="text-xs text-gray-400 mt-2">松开后隐藏</p>
                    </div>
                  </div>
                ) : (
                  /* 跟读模式（默认：展示台词）*/
                  <div>
                    <div className="flex items-start gap-3 mb-3">
                      <div className="w-10 h-10 rounded-full flex items-center justify-center shrink-0" style={{ backgroundColor: '#637FF120' }}>
                        <span className="text-lg">📝</span>
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900 mb-1.5">台词卡</h3>
                        <p className="text-gray-700 leading-relaxed text-[15px]">
                          {currentMagicSentence || <span className="text-slate-400 italic">AI 正在准备台词...</span>}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span className="px-2 py-0.5 rounded-full" style={{ backgroundColor: '#637FF115', color: '#637FF1' }}>
                        {currentScenarioTitle || '魔法重复'}
                      </span>
                      <span>•</span>
                      <span>跟读练习</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 提示 + 跳过按钮 */}
            {magicCardState !== 'passed' && (
              <div className="mt-2 flex items-center gap-2">
                <div className="flex-1 px-3 py-2 rounded-xl bg-amber-50 border border-amber-100 flex items-start gap-2">
                  <span className="text-yellow-500 text-xs mt-0.5 shrink-0">💡</span>
                  <p key={tipIndex} className="text-xs text-amber-700/80 leading-relaxed">
                    {MAGIC_TIPS[tipIndex]}
                  </p>
                </div>
                <button
                  onClick={() => {
                    if (socketRef.current?.readyState === WebSocket.OPEN) {
                      socketRef.current.send(JSON.stringify({ type: 'force_advance_magic' }));
                      console.log('⏭️ Manual advance magic task');
                    }
                  }}
                  className="shrink-0 px-3 py-2 rounded-xl text-xs text-gray-500 hover:text-gray-700 border border-gray-200 hover:border-gray-300 bg-white hover:bg-gray-50 transition-all"
                  title="跳过当前句子"
                >
                  跳过 →
                </button>
              </div>
            )}
          </div>
        )}

        {messages.map((msg, index) => {
          
          if (msg.type === 'system') {
              return (
                <div key={index} className="flex justify-center my-4">
                  <span className="text-xs text-slate-500 bg-slate-100 dark:bg-slate-800 px-3 py-1 rounded-full shadow-sm">
                    {msg.content}
                  </span>
                </div>
              );
          }
          
          const isAI = msg.type === 'ai';
          const displayContent = msg.content
            ? stripAllMarkers(msg.content.replace(/```json[\s\S]*?```/g, '')).trim()
            : '';


          if (!isAI && (!displayContent || displayContent === '...')) {
            return null;
          }
          if (isAI && !displayContent) {
            return null;
          }

          return (
            <div key={index}>
              <MessageBubble
                type={isAI ? 'ai' : 'user'}
                message={displayContent}
                state={isAI && aiBubbleRenderState({
                  isFinal: msg.isFinal,
                  hasContent: !!displayContent,
                  audioUrl: msg.audioUrl,
                  audioPlayed: msg.audioPlayed,
                }) === 'dots' ? 'loading' : 'default'}
                footer={msg.audioUrl ? (
                  <AudioBar
                    audioUrl={msg.audioUrl}
                    duration={0}
                    onClick={() => {
                      if (playingAudioUrl === msg.audioUrl) {
                        stopAudioPlayback();
                      } else {
                        playFullAudio(msg.audioUrl);
                      }
                    }}
                    isOwnMessage={!isAI}
                    isActive={playingAudioUrl === msg.audioUrl}
                  />
                ) : null}
                translation={msg.translation}
              />
              {isAI && (msg.audioUrl || msg.audioPlayed === true) && !msg.translation && (
                <div className="flex justify-start px-4 py-1">
                  <button
                    onClick={async () => {
                      try {
                        const result = await aiAPI.translate(displayContent, user?.native_language || 'zh');
                        setMessages(prev => {
                          const newMessages = [...prev];
                          newMessages[index] = { ...newMessages[index], translation: result.translation };
                          return newMessages;
                        });
                      } catch (err) {
                        console.error('Translation error:', err);
                      }
                    }}
                    aria-label="翻译这条 AI 回复"
                    className="text-xs text-slate-600 hover:text-slate-800 flex items-center gap-1 transition"
                  >
                    <span aria-hidden="true" style={{fontSize:'13px'}}>🌐</span> 翻译
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <div ref={messagesEndRef} className="h-4" />
      </main>

      {/* CC Immersive Overlay */}
      {ccMode && (
        <section
          aria-label="CC 沉浸模式"
          style={{
          position: 'absolute', inset: '56px 0 120px 0',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          background: 'radial-gradient(ellipse at top, rgba(99,127,241,0.18), transparent 60%) var(--background)',
          zIndex: 10,
          }}
        >
          <button
            onClick={() => setCcMode(false)}
            aria-label="退出 CC 沉浸模式"
            className="min-h-11 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
            style={{
            position: 'absolute', top: 10, right: 14, zIndex: 1,
            background: 'var(--card)', border: '1px solid var(--border-solid)',
            borderRadius: 20, padding: '5px 12px', fontSize: 11, fontWeight: 600,
            color: 'var(--foreground-muted)', cursor: 'pointer', fontFamily: 'inherit',
            }}
          >退出 CC <span aria-hidden="true">×</span></button>

          {/* 当前子任务进度（CC 浮层会盖住主视图任务面板，这里补一个紧凑进度指示） */}
          {!isRecallMode && !isDailyQAMode && (
            <div style={{
              position: 'absolute', top: 12, left: 0, right: 0,
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
            }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--foreground-muted)' }}>
                子任务 {Math.min(theaterCompletedTasks.size + 1, Math.max(tasks.length, 1))}/{Math.max(tasks.length, 1)}
              </span>
              <div
                role="progressbar"
                aria-label="CC 当前子任务进度"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(currentTaskProgress)}
                style={{ width: 160, height: 6, borderRadius: 3, background: 'rgba(99,127,241,0.15)', overflow: 'hidden' }}
              >
                <div style={{
                  width: `${Math.max(0, Math.min(100, currentTaskProgress))}%`,
                  height: '100%', borderRadius: 3, background: '#637FF1',
                  transition: 'width 0.4s ease',
                }} />
              </div>
              {!isTourMode && currentPhase === 'scene_theater' && activeScoringTask
                && currentTaskScore >= 9 && currentTaskProgress < 100 && (
                <div className="w-full max-w-lg max-h-[25vh] overflow-y-auto px-4">
                  <TaskProgressGuidance
                    feedback={progressFeedback && acceptScoringMessage(progressFeedback) ? progressFeedback : null}
                    taskTitle={activeScoringTask.text}
                    ready={Boolean(taskReadyToComplete && acceptScoringMessage(taskReadyToComplete))}
                    onConfirm={() => setCompletionSheetDismissed(false)}
                  />
                </div>
              )}
            </div>
          )}

          <GuajiMascot state={avatarStatus} size={200} />
          <CCRollingCaption
            isAISpeaking={isAISpeaking}
            text={(() => {
              const lastAI = [...messages].reverse().find(m => m.type === 'ai' && m.content);
              return (lastAI?.content || '').trim();
            })()}
            getProgressRatio={() => {
              const ctx = audioContextRef.current;
              if (!ctx) return 0;
              const total = speechTotalDurationRef.current;
              if (total <= 0) return 0;
              const elapsed = ctx.currentTime - speechStartTimeRef.current;
              return Math.max(0, Math.min(1, elapsed / total));
            }}
          />
        </section>
      )}

      {/* Footer / Controls */}
      <footer className="pb-4 pt-3 px-4 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 shrink-0 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
        <div className="flex flex-col items-center gap-3">
            {/* Main Controls: Recorder + Restart Button */}
            <div className="flex items-center gap-3 w-full max-w-md" data-testid="conversation-footer-controls">
                <div className="flex-1 min-w-0 relative" data-tour="mic">
                    {/* 3-场景仅为软性鼓励，不再禁用录音；硬护栏由后端 daily_limit_reached 负责 */}
                    <RealTimeRecorder
                      ref={recorderRef}
                      isConnected={isConnected}
                      onStart={handleRecordingStart}
                      onStop={handleRecordingStop}
                      onCancel={handleRecordingCancel}
                      enableCompression={true}
                      enableMetrics={true}
                    />
                </div>
                
                {/* Keep the toggle in place while active so the footer width does
                    not reflow and clip the restart action on narrow screens. */}
                <button
                  data-tour="cc-mode"
                  onClick={() => setCcMode(value => !value)}
                  aria-label={ccMode ? '退出 CC 沉浸模式' : '进入 CC 沉浸模式'}
                  aria-pressed={ccMode}
                  className={`${isUserRecording ? 'hidden sm:flex' : 'flex'} flex-shrink-0 w-12 h-12 rounded-xl items-center justify-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50`}
                  style={{
                    border: `1.5px solid ${ccMode ? 'var(--primary)' : 'var(--border-solid)'}`,
                    background: ccMode ? 'rgba(99,127,241,0.12)' : 'var(--card)',
                    color: ccMode ? 'var(--primary-dark)' : 'var(--foreground-muted)',
                    fontSize: 13, fontWeight: 700,
                  }}
                  title={ccMode ? '退出沉浸模式' : '进入沉浸模式'}
                >CC</button>

                {/* Restart Practice Button - Icon only */}
                {(tasks.length > 0 || location.state?.scenario || new URLSearchParams(window.location.search).get('scenario')) && (
                    <button
                      onClick={() => {
                        // 二次确认：重置会清空当前对话与进度，不可撤销，防误触
                        if (window.confirm('确定重新练习？当前对话进度将被清空，且无法恢复。')) {
                          handleRetryCurrentScenario({ keepHistory: false, resetProgress: true });
                        }
                      }}
                      className={`${isUserRecording ? 'hidden sm:flex' : 'flex'} flex-shrink-0 w-12 h-12 bg-amber-100 dark:bg-amber-900/30 hover:bg-amber-200 dark:hover:bg-amber-900/50 text-amber-700 dark:text-amber-300 rounded-xl items-center justify-center transition border border-amber-200 dark:border-amber-700`}
                      aria-label="重新练习当前场景"
                      title="重新练习"
                    >
                      <span className="material-symbols-outlined" aria-hidden="true">replay</span>
                    </button>
                )}
            </div>

            {/* WebSocket Error Display with Retry / Back exit */}
            {shouldShowConnectionError(webSocketError, isConnected, wsRejected) && (
                <div className="flex items-center gap-3 w-full max-w-md">
                    <p className="text-xs text-red-500 bg-red-50 dark:bg-red-900/20 px-3 py-1.5 rounded-full flex-1">
                        {webSocketError}
                    </p>

                    {wsRejected ? (
                        /* Backend rejected this connection (e.g. invalid scenario) —
                           retrying loops the same rejection, so offer an exit instead. */
                        <button
                            onClick={() => navigate('/discovery')}
                            className="px-3 py-1.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-xs font-medium flex items-center gap-1 hover:bg-red-200 dark:hover:bg-red-900/50 transition"
                        >
                            <span className="material-symbols-outlined text-sm">arrow_back</span>
                            <span>{t('ws_error_back_to_discovery', '返回发现页')}</span>
                        </button>
                    ) : (
                        /* Retry Button - show when connection fails or max attempts reached */
                        <button
                            onClick={handleManualRetry}
                            className="px-3 py-1.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-xs font-medium flex items-center gap-1 hover:bg-red-200 dark:hover:bg-red-900/50 transition animate-pulse"
                        >
                            <span className="material-symbols-outlined text-sm">refresh</span>
                            <span>重试</span>
                        </button>
                    )}
                </div>
            )}
        </div>
      </footer>

      {/* AI Feedback Strip */}
      <AIFeedbackStrip />

      {/* Scenario Completion Modal */}
      {showCompletionModal && (
        <PracticeReport
          scenarioTitle={currentScenarioTitle}
          scenarioScore={scenarioScore}
          reviewData={scenarioReviewData}
          messages={messages}
          onClose={() => {
            setShowCompletionModal(false);
            hasViewedCompletionModalRef.current = true;
          }}
          onRestart={() => handleRetryCurrentScenario({ keepHistory: false, resetProgress: true })}
          onNextScenario={handleNextScenario}
          onSelectOther={handleSelectOtherScenario}
          hasNextScenario={currentScenarioIndex < allScenarios.length - 1}
          onCheckin={() => userAPI.checkin()}
        />
      )}
      {showScorePopup && batchScores && (
        <ScorePopup
          scores={batchScores}
          delta={latestDelta}
          onClose={() => setShowScorePopup(false)}
        />
      )}
      {showDailyQAPassModal && (
        <DailyQAPassModal
          onClose={() => setShowDailyQAPassModal(false)}
          onReturn={() => navigate('/discovery')}
          isBonus={dailyQAIsBonus}
        />
      )}
      {dailyLimitModal && (
        <AccessibleDialog
          title={dailyLimitModal.kind === 'paywall' ? t('daily_limit_paywall_title') : t('daily_limit_reached_title')}
          description={dailyLimitModal.kind === 'paywall'
            ? t('daily_limit_paywall_desc', { limit: dailyLimitModal.limit })
            : t('daily_limit_reached_desc', { limit: dailyLimitModal.limit })}
          onClose={() => setDailyLimitModal(null)}
          closeLabel={t('daily_limit_cancel')}
          showCloseButton={false}
          panelClassName="!max-w-[360px] !rounded-3xl"
          zIndex={300}
        >
          <div
            style={{
              padding: 32, textAlign: 'center',
            }}>
            <div aria-hidden="true" style={{ fontSize: 56, marginBottom: 12 }}>
              {dailyLimitModal.kind === 'paywall' ? '🔒' : '🌙'}
            </div>
            <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1F2937', marginBottom: 8 }}>
              {dailyLimitModal.kind === 'paywall' ? t('daily_limit_paywall_title') : t('daily_limit_reached_title')}
            </h2>
            <p style={{ fontSize: 14, color: '#6B7280', marginBottom: 24, lineHeight: 1.55 }}>
              {dailyLimitModal.kind === 'paywall'
                ? t('daily_limit_paywall_desc', { limit: dailyLimitModal.limit })
                : t('daily_limit_reached_desc', { limit: dailyLimitModal.limit })}
            </p>
            <div style={{ display: 'flex', gap: 12 }}>
              <button
                onClick={() => setDailyLimitModal(null)}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 12,
                  background: '#F3F4F6', color: '#374151', border: 'none',
                  fontWeight: 600, fontSize: 14, cursor: 'pointer',
                }}>
                {dailyLimitModal.kind === 'paywall' ? t('daily_limit_cancel') : t('daily_limit_got_it')}
              </button>
              {dailyLimitModal.ctaToSubscription && (
                <button
                  onClick={() => navigate('/subscription')}
                  style={{
                    flex: 1, padding: '10px 0', borderRadius: 12,
                    background: '#6366F1', color: '#fff', border: 'none',
                    fontWeight: 700, fontSize: 14, cursor: 'pointer',
                  }}>
                  {t('daily_limit_upgrade')}
                </button>
              )}
            </div>
          </div>
        </AccessibleDialog>
      )}
      {/* Language gate warning — visible amber banner anchored to the top.
          Without this, the only signal that the daily QA was rejected for
          wrong language was a server log; users mistakenly thought their
          Chinese answer to an English question had passed. */}
      <AnimatePresence>
        {languageGateWarning && (
          <motion.div
            initial={{ y: -40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -40, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 380, damping: 28 }}
            style={{
              position: 'fixed', top: 12, left: '50%', transform: 'translateX(-50%)',
              maxWidth: '92%', zIndex: 400,
              background: 'linear-gradient(135deg, #F59E0B, #D97706)', color: '#fff',
              padding: '12px 18px', borderRadius: 14,
              boxShadow: '0 10px 30px rgba(217,119,6,0.35)',
              fontSize: 14, fontWeight: 600,
              display: 'flex', alignItems: 'flex-start', gap: 10,
            }}
            role="alert"
          >
            <span style={{ fontSize: 20, lineHeight: 1 }}>⚠️</span>
            <span style={{ flex: 1, lineHeight: 1.45 }}>{languageGateWarning.message}</span>
            <button
              onClick={() => setLanguageGateWarning(null)}
              aria-label="关闭提示"
              style={{
                background: 'rgba(255,255,255,0.2)', border: 'none', color: '#fff',
                width: 22, height: 22, borderRadius: 11, cursor: 'pointer',
                fontWeight: 700, lineHeight: 1, fontSize: 13,
              }}
            >×</button>
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {taskReadyToComplete && acceptScoringMessage(taskReadyToComplete) && !completionSheetDismissed && (
          <TaskCompletionSheet
            taskReadyToComplete={taskReadyToComplete}
            tasks={tasks}
            completedTasks={completedTasks}
            onConfirm={handleConfirmComplete}
            onContinue={() => {
              setTaskCompletionPending(false);
              setCompletionSheetDismissed(true);
            }}
            canConfirm={isConnected && !taskCompletionPending}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default Conversation;
