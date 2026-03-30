import { Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { overviewApi, type OverviewResponse } from '../api';
import MetricCard from '../components/MetricCard';
import EmptyState from '../components/EmptyState';
import { formatDate, formatInstruction, formatPercent, truncate } from '../utils/format';

export default function DashboardPage() {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const overview = await overviewApi.getOverview();
        if (!cancelled) {
          setData(overview);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load overview');
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

  if (loading) {
    return <div className="p-6 text-sm text-muted">Loading dashboard…</div>;
  }

  if (error || !data) {
    return <div className="p-6 text-sm text-danger">{error ?? 'Dashboard unavailable'}</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold font-serif">Dashboard</h1>
      <section className="metrics-row">
        <MetricCard label="Total Skills" value={data.skills.summary.total_skills_all} hint={`Active: ${data.skills.summary.total_skills}`} />
        <MetricCard label="Average Skill Score" value={data.skills.average_score.toFixed(1)} hint="Primary metric = effective rate × 100" />
        <MetricCard label="Workflow Sessions" value={data.workflows.total} hint={`Recorded under ${data.health.db_path.includes('.openspace') ? 'local repo' : 'workspace'}`} />
        <MetricCard label="Workflow Success" value={`${data.workflows.average_success_rate.toFixed(1)}%`} hint="Average session success rate" />
      </section>

      <section>
        <div className="panel-surface p-5 space-y-4">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-muted">Health</div>
            <h2 className="text-2xl font-bold font-serif mt-1">Runtime snapshot</h2>
          </div>
          <div className="space-y-3 text-sm">
            <div className="flex items-center justify-between"><span className="text-muted">Status</span><span>{data.health.status}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">DB Path</span><span className="text-right break-all">{data.health.db_path}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">Workflow Count</span><span>{data.health.workflow_count}</span></div>
            <div className="flex items-center justify-between"><span className="text-muted">Built Frontend</span><span>{data.health.frontend_dist_exists ? 'yes' : 'no'}</span></div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-6">
        <div className="panel-surface p-5 space-y-4">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-muted">Skills</div>
            <h2 className="text-2xl font-bold font-serif mt-1">Top scored skills</h2>
          </div>
          {data.skills.top.length === 0 ? (
            <EmptyState title="No skills yet" description="Run OpenSpace tasks or sync skills into the local registry first." />
          ) : (
            <div className="space-y-3">
              {data.skills.top.map((skill) => (
                <Link key={skill.skill_id} to={`/skills/${encodeURIComponent(skill.skill_id)}`} className="record-card block p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 space-y-1">
                      <div className="font-bold truncate">{skill.name}</div>
                      <div className="text-sm text-muted">{truncate(skill.description || 'No description', 110)}</div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-2xl font-bold font-serif">{skill.score.toFixed(1)}</div>
                      <div className="text-xs text-muted">score</div>
                    </div>
                  </div>
                  <div className="mt-3 flex gap-3 text-xs text-muted">
                    <span>effective {formatPercent(skill.effective_rate)}</span>
                    <span>applied {formatPercent(skill.applied_rate)}</span>
                    <span>selections {skill.total_selections}</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        <div className="panel-surface p-5 space-y-4">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-muted">Workflows</div>
            <h2 className="text-2xl font-bold font-serif mt-1">Recent sessions</h2>
          </div>
          {data.workflows.recent.length === 0 ? (
            <EmptyState title="No workflow sessions" description="Recordings will appear after a task is executed with recording enabled." />
          ) : (
            <div className="space-y-3">
              {data.workflows.recent.map((workflow) => (
                <Link key={workflow.id} to={`/workflows/${encodeURIComponent(workflow.id)}`} className="record-card block p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 space-y-1">
                      <div className="font-bold truncate">{workflow.task_name}</div>
                      <div className="text-sm text-muted line-clamp-2">{formatInstruction(workflow.instruction, 160)}</div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-lg font-bold font-serif">{(workflow.success_rate * 100).toFixed(1)}%</div>
                      <div className="text-xs text-muted">success</div>
                    </div>
                  </div>
                  <div className="mt-3 flex gap-3 text-xs text-muted">
                    <span>{workflow.total_steps} steps</span>
                    <span>{workflow.agent_action_count} agent actions</span>
                    <span>{formatDate(workflow.start_time)}</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
