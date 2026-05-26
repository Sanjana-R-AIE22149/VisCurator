/**
 * PipelineWizard — persistent 4-step progress banner
 * Shown whenever a job is active (jobId set in the store).
 * Steps: Fetch → Annotate → Augment → Train
 * Clicking a step navigates to that page.
 */
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppStore } from '../../store/useAppStore';
import { Database, Layers, Wand2, Cpu, ChevronRight, CheckCircle2, Loader2, Circle } from 'lucide-react';

type StepId = 'dataset' | 'annotator' | 'augmentation' | 'builder';

interface Step {
  id: StepId;
  label: string;
  icon: React.ElementType;
  path: string;
}

const STEPS: Step[] = [
  { id: 'dataset',     label: 'Fetch',     icon: Database, path: '/dataset'     },
  { id: 'annotator',  label: 'Annotate',   icon: Layers,   path: '/annotator'   },
  { id: 'augmentation', label: 'Augment',  icon: Wand2,    path: '/augmentation' },
  { id: 'builder',    label: 'Train',      icon: Cpu,      path: '/builder'     },
];

type StepStatus = 'done' | 'active' | 'pending';

function deriveStatuses(
  pathname: string,
  isProcessing: boolean,
  preprocessingReport: unknown,
  annoReport: boolean,
  isTraining: boolean,
  trainingRunId: string | null,
): Record<StepId, StepStatus> {
  // Determine farthest completed step
  const hasFetch  = !!preprocessingReport;
  const hasAnnot  = annoReport;
  const hasAug    = false; // future: track aug completion in store
  const hasTrain  = !!trainingRunId;

  const seg = pathname.replace('/', '').split('/')[0] as StepId | '';

  return {
    dataset:      hasFetch && seg !== 'dataset'     ? 'done'    : seg === 'dataset'     ? 'active' : 'pending',
    annotator:    hasAnnot && seg !== 'annotator'   ? 'done'    : seg === 'annotator'   ? 'active' : hasFetch   ? 'pending' : 'pending',
    augmentation: hasAug  && seg !== 'augmentation' ? 'done'    : seg === 'augmentation'? 'active' : hasAnnot   ? 'pending' : 'pending',
    builder:      hasTrain && !isTraining           ? 'done'    : isTraining || seg === 'builder' ? 'active' : 'pending',
  };
}

const STATUS_STYLE: Record<StepStatus, { ring: string; icon: string; text: string; bg: string }> = {
  done:    { ring: 'border-teal-500/60',   icon: 'text-teal-400',   text: 'text-teal-300',   bg: 'bg-teal-500/10'   },
  active:  { ring: 'border-violet-500/70', icon: 'text-violet-400', text: 'text-violet-300', bg: 'bg-violet-500/10' },
  pending: { ring: 'border-slate-700/50',  icon: 'text-slate-600',  text: 'text-slate-600',  bg: 'bg-transparent'   },
};

export default function PipelineWizard() {
  const { jobId, isProcessing, preprocessingReport, isTraining, trainingRunId } = useAppStore();
  const { pathname } = useLocation();
  const navigate = useNavigate();

  // Only show when a pipeline job is active
  if (!jobId) return null;

  const statuses = deriveStatuses(
    pathname, isProcessing, preprocessingReport, false, isTraining, trainingRunId,
  );

  return (
    <div
      id="pipeline-wizard"
      className="flex items-center justify-center gap-1 px-4 py-2 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-sm"
    >
      {STEPS.map((step, i) => {
        const status = statuses[step.id];
        const style  = STATUS_STYLE[status];
        const Icon   = step.icon;
        const isLast = i === STEPS.length - 1;

        return (
          <div key={step.id} className="flex items-center gap-1">
            <button
              id={`wizard-step-${step.id}`}
              onClick={() => navigate(step.path)}
              title={`Go to ${step.label}`}
              className={`
                flex items-center gap-1.5 px-3 py-1 rounded-full border text-[11px] font-semibold
                transition-all duration-200 hover:scale-105
                ${style.ring} ${style.bg} ${style.text}
              `}
            >
              {status === 'done' ? (
                <CheckCircle2 className={`w-3 h-3 ${style.icon}`} />
              ) : status === 'active' ? (
                isProcessing || isTraining
                  ? <Loader2 className={`w-3 h-3 ${style.icon} animate-spin`} />
                  : <Icon className={`w-3 h-3 ${style.icon}`} />
              ) : (
                <Circle className={`w-3 h-3 ${style.icon}`} />
              )}
              {step.label}
            </button>

            {!isLast && (
              <ChevronRight className="w-3 h-3 text-slate-700 shrink-0" />
            )}
          </div>
        );
      })}

      {/* Active job chip */}
      <div className="ml-3 pl-3 border-l border-slate-800 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
        <span className="text-[10px] font-mono text-slate-600 max-w-[120px] truncate">
          {jobId.slice(0, 8)}…
        </span>
      </div>
    </div>
  );
}
