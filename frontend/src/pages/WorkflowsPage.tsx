import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { workflowsApi, type WorkflowSummary } from '../api';
import EmptyState from '../components/EmptyState';
import { formatDate, formatInstruction } from '../utils/format';

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const items = await workflowsApi.listWorkflows();
        if (!cancelled) {
          setWorkflows(items);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load workflows');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const list = normalized
      ? workflows.filter((workflow) =>
          workflow.task_name.toLowerCase().includes(normalized) ||
          workflow.task_id.toLowerCase().includes(normalized) ||
          workflow.instruction.toLowerCase().includes(normalized),
        )
      : [...workflows];
    return list.sort((a, b) => new Date(b.start_time ?? 0).getTime() - new Date(a.start_time ?? 0).getTime());
  }, [query, workflows]);

  const averageSuccess = workflows.length > 0
    ? ((workflows.reduce((sum, item) => sum + item.success_rate, 0) / workflows.length) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold font-serif">Recorded sessions</h1>
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by task name or instruction"
          className="px-3 py-2 min-w-[320px]"
        />
      </div>

      <section className="grid grid-cols-2 gap-4">
        <div className="p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.16em] text-muted">Workflow Sessions</div>
          <div className="text-3xl font-bold font-serif leading-none">{workflows.length}</div>
          <div className="text-xs text-muted">Scanned from logs/recordings and logs/trajectories</div>
        </div>
        <div className="p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.16em] text-muted">Average Success</div>
          <div className="text-3xl font-bold font-serif leading-none">{averageSuccess}%</div>
          <div className="text-xs text-muted">Mean success rate across sessions</div>
        </div>
      </section>

      {loading ? <div className="text-sm text-muted">Loading workflows…</div> : null}
      {error ? <div className="text-sm text-danger">{error}</div> : null}

      {!loading && !error && filtered.length === 0 ? (
        <EmptyState title="No workflow sessions" description="Run `openspace` with recording enabled, then refresh this page." />
      ) : null}

      {!loading && !error && filtered.length > 0 ? (
        <div className="grid grid-cols-2 gap-4">
          {filtered.map((workflow) => (
            <Link key={workflow.id} to={`/workflows/${encodeURIComponent(workflow.id)}`} className="record-card bg-surface block p-4 space-y-3 hover:border-primary transition-colors">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="font-bold truncate">{workflow.task_name}</div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-lg font-bold font-serif">{(workflow.success_rate * 100).toFixed(1)}%</div>
                  <div className="text-xs text-muted">success</div>
                </div>
              </div>
              <div className="text-sm text-muted line-clamp-2">{formatInstruction(workflow.instruction, 220)}</div>
              <div className="grid grid-cols-3 gap-3 text-xs text-muted">
                <div>{workflow.total_steps} steps</div>
                <div>{workflow.agent_action_count} agent actions</div>
                <div>{formatDate(workflow.start_time)}</div>
              </div>
              {workflow.selected_skills.length > 0 ? (
                <div className="flex flex-wrap gap-2 text-xs">
                  {workflow.selected_skills.slice(0, 3).map((skillId, index) => (
                    <span key={`${skillId}-${index}`} title={skillId} className="tag inline-flex max-w-full items-center px-2 py-1">
                      <span className="block max-w-[220px] truncate">{skillId}</span>
                    </span>
                  ))}
                  {workflow.selected_skills.length > 3 ? (
                    <span className="tag px-2 py-1 text-muted">
                      +{workflow.selected_skills.length - 3} more
                    </span>
                  ) : null}
                </div>
              ) : null}
            </Link>
          ))}
        </div>
      ) : null}
    </div>
  );
}
