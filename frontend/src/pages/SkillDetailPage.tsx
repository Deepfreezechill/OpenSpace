import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams, useParams } from 'react-router-dom';
import { skillsApi, type SkillDetail, type SkillLineage } from '../api';
import EmptyState from '../components/EmptyState';
import MetricCard from '../components/MetricCard';
import SkillEvolutionGraph from '../components/skill-detail/SkillEvolutionGraph';
import SkillVersionDrawer from '../components/skill-detail/SkillVersionDrawer';
import SkillVersionFilterBar from '../components/skill-detail/SkillVersionFilterBar';
import { useSkillEvolutionGraphData } from '../hooks/useSkillEvolutionGraphData';
import { formatDate } from '../utils/format';

function resolveLineageGraph(skill: SkillDetail | null): SkillLineage | null {
  if (!skill) {
    return null;
  }
  if (skill.lineage_graph && Array.isArray(skill.lineage_graph.nodes)) {
    return skill.lineage_graph;
  }
  const legacyGraph = (skill as SkillDetail & { lineage?: SkillLineage }).lineage;
  if (legacyGraph && Array.isArray(legacyGraph.nodes)) {
    return legacyGraph;
  }
  return null;
}

const DRAWER_ANIMATION_DURATION_MS = 300;

export default function SkillDetailPage() {
  const { skillId = '' } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [skillClass, setSkillClass] = useState<SkillDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<SkillDetail | null>(null);
  const [drawerVersion, setDrawerVersion] = useState<SkillDetail | null>(null);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [drawerError, setDrawerError] = useState<string | null>(null);
  const [originFilter, setOriginFilter] = useState('all');
  const [tagFilter, setTagFilter] = useState('all');

  const selectedVersionId = searchParams.get('version');

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const detail = await skillsApi.getSkill(skillId);
        if (!cancelled) {
          setSkillClass(detail);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load skill class');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    if (skillId) {
      void load();
    }

    return () => {
      cancelled = true;
    };
  }, [skillId]);

  const lineageGraph = useMemo(() => resolveLineageGraph(skillClass), [skillClass]);

  useEffect(() => {
    if (!selectedVersionId) {
      setSelectedVersion(null);
      setDrawerError(null);
      return;
    }

    if (skillClass && selectedVersionId === skillClass.skill_id) {
      setSelectedVersion(skillClass);
      setDrawerError(null);
      return;
    }

    let cancelled = false;
    const loadSelectedVersion = async () => {
      setDrawerLoading(true);
      setDrawerError(null);
      try {
        const detail = await skillsApi.getSkill(selectedVersionId);
        if (!cancelled) {
          setSelectedVersion(detail);
        }
      } catch (err) {
        if (!cancelled) {
          setSelectedVersion(null);
          setDrawerError(err instanceof Error ? err.message : 'Failed to load selected version');
        }
      } finally {
        if (!cancelled) {
          setDrawerLoading(false);
        }
      }
    };

    void loadSelectedVersion();
    return () => {
      cancelled = true;
    };
  }, [selectedVersionId, skillClass]);

  useEffect(() => {
    if (selectedVersion) {
      setDrawerVersion(selectedVersion);
      return;
    }

    if (!drawerVersion) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setDrawerVersion(null);
    }, DRAWER_ANIMATION_DURATION_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [drawerVersion, selectedVersion]);

  useEffect(() => {
    if (!lineageGraph || !selectedVersionId) {
      return;
    }
    const exists = lineageGraph.nodes.some((node) => node.skill_id === selectedVersionId);
    if (!exists && skillClass && selectedVersionId !== skillClass.skill_id) {
      const next = new URLSearchParams(searchParams);
      next.delete('version');
      setSearchParams(next);
    }
  }, [lineageGraph, searchParams, selectedVersionId, setSearchParams, skillClass]);

  const alwaysVisibleSkillIds = useMemo(
    () => [skillId, selectedVersionId].filter((value): value is string => Boolean(value)),
    [selectedVersionId, skillId],
  );

  const { allOrigins, allTags, graphData } = useSkillEvolutionGraphData(
    lineageGraph,
    originFilter,
    tagFilter,
    alwaysVisibleSkillIds,
  );

  const classSummary = useMemo(() => {
    const nodes = lineageGraph?.nodes ?? [];
    if (nodes.length === 0) {
      return {
        versionCount: 0,
        activeCount: 0,
        bestScore: 0,
        averageScore: 0,
        maxGeneration: 0,
        totalSelections: 0,
        latestCreatedAt: null as string | null,
        tags: [] as string[],
        origins: [] as string[],
      };
    }

    const tags = new Set<string>();
    const origins = new Set<string>();
    let totalSelections = 0;
    let bestScore = 0;
    let maxGeneration = 0;
    let latestCreatedAt: string | null = null;

    nodes.forEach((node) => {
      node.tags.forEach((tag) => tags.add(tag));
      origins.add(node.origin);
      totalSelections += node.total_selections;
      bestScore = Math.max(bestScore, node.score);
      maxGeneration = Math.max(maxGeneration, node.generation);
      if (!latestCreatedAt || Date.parse(node.created_at) > Date.parse(latestCreatedAt)) {
        latestCreatedAt = node.created_at;
      }
    });

    return {
      versionCount: nodes.length,
      activeCount: nodes.filter((node) => node.is_active).length,
      bestScore,
      averageScore: nodes.reduce((sum, node) => sum + node.score, 0) / nodes.length,
      maxGeneration,
      totalSelections,
      latestCreatedAt,
      tags: Array.from(tags).sort(),
      origins: Array.from(origins).sort(),
    };
  }, [lineageGraph]);

  const openVersion = (nextSkillId: string) => {
    const next = new URLSearchParams(searchParams);
    next.set('version', nextSkillId);
    setSearchParams(next);
  };

  const closeDrawer = () => {
    const next = new URLSearchParams(searchParams);
    next.delete('version');
    setSearchParams(next);
  };

  if (loading) {
    return <div className="p-6 text-sm text-muted">Loading skill detail…</div>;
  }

  if (error || !skillClass) {
    return <div className="p-6 text-sm text-danger">{error ?? 'Skill not found'}</div>;
  }

  return (
    <div className="p-6 space-y-6 relative">
      <div className="flex items-center gap-4">
        <Link to="/skills" className="chip text-sm transition-colors hover:border-[color:var(--color-border-dark)] hover:text-ink">← Back to Skills</Link>
        <div className="min-w-0">
          <h1 className="text-3xl font-bold font-serif truncate">{skillClass.name}</h1>
          <div className="text-sm text-muted mt-1">Skill class anchored on {skillClass.skill_id}</div>
        </div>
      </div>

      <section className="panel-surface p-5 space-y-4">
        <div className="flex items-start justify-between gap-6">
          <div className="space-y-3 min-w-0 flex-1">
            <div>
              <div className="text-xs uppercase tracking-[0.16em] text-muted">Skill Class</div>
              <h2 className="text-2xl font-bold font-serif mt-1">Evolution overview</h2>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="tag px-2 py-1">{skillClass.category}</span>
              <span className="tag px-2 py-1">{skillClass.visibility}</span>
              <span className="tag px-2 py-1">{skillClass.is_active ? 'active tip' : 'inactive anchor'}</span>
              {classSummary.origins.map((origin) => (
                <span key={origin} className="tag px-2 py-1">{origin}</span>
              ))}
              {classSummary.tags.slice(0, 8).map((tag) => (
                <span key={tag} className="tag px-2 py-1">{tag}</span>
              ))}
              {classSummary.tags.length > 8 ? (
                <span className="tag px-2 py-1">+{classSummary.tags.length - 8} tags</span>
              ) : null}
            </div>
          </div>
          <div className="shrink-0 text-right">
            <div className="text-5xl font-bold font-serif leading-none">{classSummary.bestScore.toFixed(1)}</div>
            <div className="text-xs uppercase tracking-[0.16em] text-muted mt-2">best version score</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm text-muted">
          <div>
            <div className="font-bold text-ink">Skill directory</div>
            <div className="break-all">{skillClass.skill_dir || 'Unavailable'}</div>
          </div>
          <div>
            <div className="font-bold text-ink">Latest version created</div>
            <div>{formatDate(classSummary.latestCreatedAt)}</div>
          </div>
          <div>
            <div className="font-bold text-ink">Representative version</div>
            <div className="break-all">{skillClass.skill_id}</div>
          </div>
          <div>
            <div className="font-bold text-ink">Representative update</div>
            <div>{formatDate(skillClass.last_updated)}</div>
          </div>
        </div>
      </section>

      <section className="metrics-row">
        <MetricCard label="Versions" value={classSummary.versionCount} hint={`Max generation ${classSummary.maxGeneration}`} />
        <MetricCard label="Active Versions" value={classSummary.activeCount} hint={`Origins: ${classSummary.origins.length}`} />
        <MetricCard label="Average Score" value={classSummary.averageScore.toFixed(1)} hint="Across all versions in this lineage" />
        <MetricCard label="Selections" value={classSummary.totalSelections} hint={`Representative score ${skillClass.score.toFixed(1)}`} />
      </section>

      <section className="panel-surface overflow-hidden relative min-h-[620px]">
        <div className="px-5 py-4 border-b border-[color:var(--color-border)] bg-surface flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-muted">Evolution Graph</div>
            <h2 className="text-2xl font-bold font-serif mt-1">Version lineage</h2>
          </div>
          <SkillVersionFilterBar
            originFilter={originFilter}
            onOriginFilterChange={setOriginFilter}
            tagFilter={tagFilter}
            onTagFilterChange={setTagFilter}
            allOrigins={allOrigins}
            allTags={allTags}
          />
        </div>
        <SkillEvolutionGraph
          graphData={graphData}
          selectedNodeId={selectedVersionId}
          onNodeClick={(node) => openVersion(node.id)}
          onBackgroundClick={closeDrawer}
        />
        {drawerLoading ? (
          <div className="absolute bottom-4 left-4 text-xs text-muted">Loading version drawer…</div>
        ) : null}
        {drawerError ? (
          <div className="absolute bottom-4 left-4 text-xs text-danger">{drawerError}</div>
        ) : null}
      </section>

      {lineageGraph && lineageGraph.nodes.length === 0 ? (
        <EmptyState title="No lineage graph" description="This skill does not yet have lineage data to visualize." />
      ) : null}

      <SkillVersionDrawer skill={drawerVersion} isOpen={Boolean(selectedVersion)} onClose={closeDrawer} />
    </div>
  );
}
