'use client'
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/Card';
import { Progress } from './ui/progress';
import { 
  Gavel, 
  Lock, 
  Eye, 
  Clock, 
  ArrowRight, 
  Shield, 
  Zap, 
  RefreshCw,
  CheckCircle2
} from 'lucide-react';

interface TimelinePhase {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  details: string[];
  icon: React.ReactNode;
  color: string;
  status: 'completed' | 'active' | 'pending';
}

const FusionTimeline: React.FC = () => {
  const [activePhase, setActivePhase] = useState<string>('visualization');
  const [animationKey, setAnimationKey] = useState(0);

  const phases: TimelinePhase[] = [
    {
      id: 'visualization',
      title: 'N-Dimensional Sphere',
      subtitle: 'Reserve Vectors in Space',
      description: 'Orbital treats AMM reserves as vectors in n-dimensional space, where each token represents a dimension and reserves form a point on a spherical surface.',
      details: [
        'Reserves represented as vector r = (x₁, x₂, ..., xₙ)',
        'Spherical constraint Σ(xᵢ - r)² = r² maintains constant sum',
        'Equal price point where all reserves are equal',
        'Geometric intuition enables visual understanding',
        'Scalable to any number of stablecoins'
      ],
      icon: <Gavel className="w-6 h-6" />,
      color: 'from-blue-500 to-cyan-500',
      status: activePhase === 'visualization' ? 'active' : 'completed'
    },
    {
      id: 'ticks',
      title: 'Orbital Tick System',
      subtitle: 'Spherical Caps as Liquidity Ranges',
      description: 'Liquidity providers create concentrated positions using spherical caps around the equal price point, enabling customizable exposure and capital efficiency.',
      details: [
        'Ticks defined as spherical caps around equal price point',
        'Nested structure where larger ticks contain smaller ones',
        'Interior ticks behave like spherical AMMs',
        'Boundary ticks act as lower-dimensional spheres',
        'Liquidity concentration maximizes capital efficiency'
      ],
      icon: <Lock className="w-6 h-6" />,
      color: 'from-cyan-500 to-blue-500',
      status: activePhase === 'ticks' ? 'active' : activePhase === 'visualization' ? 'pending' : 'completed'
    },
    {
      id: 'mathematics',
      title: 'Spherical Invariant',
      subtitle: 'K = Σ(xᵢ - r)² Mathematical Foundation',
      description: 'The core invariant K = Σ(xᵢ - r)² maintains constant sum of squared reserves, enabling precise trade calculations with minimal slippage.',
      details: [
        'Trade calculation preserves spherical constraint',
        'Newton\'s method for integer square root precision',
        'Implementation of Q96X48 to increase precision'
      ],
      icon: <Eye className="w-6 h-6" />,
      color: 'from-blue-600 to-cyan-400',
      status: activePhase === 'mathematics' ? 'active' : ['visualization', 'ticks'].includes(activePhase) ? 'pending' : 'completed'
    },
    {
      id: 'efficiency',
      title: 'Capital Efficiency',
      subtitle: 'Concentrated Liquidity Benefits',
      description: 'By focusing liquidity where normal trading occurs and providing safety mechanisms for depeg events, Orbital achieves unprecedented capital efficiency.',
      details: [
        'Multiple times more capital efficiency vs uniform distribution',
        'Smaller ticks focus on normal trading ranges',
        'Boundary ticks provide depeg protection',
        'Customizable exposure for different LP strategies',
        'Gas optimization through integer arithmetic'
      ],
      icon: <RefreshCw className="w-6 h-6" />,
      color: 'from-cyan-600 to-blue-600',
      status: activePhase === 'efficiency' ? 'active' : 'pending'
    }
  ];

  const handlePhaseClick = (phaseId: string) => {
    setActivePhase(phaseId);
    setAnimationKey(prev => prev + 1);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-blue-400" />;
      case 'active':
        return <Zap className="w-5 h-5 text-cyan-400 animate-pulse" />;
      default:
        return <Clock className="w-5 h-5 text-neutral-400" />;
    }
  };

  const getProgressValue = () => {
    const phaseIndex = phases.findIndex(p => p.id === activePhase);
    return ((phaseIndex + 1) / phases.length) * 100;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-8 bg-black">
      <div className="text-center">
        <h2 className="text-3xl md:text-4xl font-black mb-4 bg-white bg-clip-text text-transparent font-[family-name:var(--font-unbounded)]">
          How Orbital Works
        </h2>
        <p className="text-lg text-neutral-300 max-w-3xl mx-auto mb-6 font-[family-name:var(--font-spline-sans-mono)]">
          Revolutionary <span className="text-blue-400 font-semibold">AMM mechanism</span> using 
          spherical geometry and orbital mechanics for optimal capital efficiency
        </p>
        
        <div className="max-w-xl mx-auto mb-8">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-neutral-400 font-[family-name:var(--font-spline-sans-mono)]">Progress</span>
            <span className="text-xs font-medium text-blue-400 font-[family-name:var(--font-spline-sans-mono)]">
              {Math.round(getProgressValue())}% Complete
            </span>
          </div>
          <Progress value={getProgressValue()} className="h-2" />
        </div>
      </div>

      <div className="grid md:grid-cols-4 gap-3 mb-8">
                  {phases.map((phase) => (
          <motion.button
            key={phase.id}
            onClick={() => handlePhaseClick(phase.id)}
            className={`p-3 rounded-xl text-left transition-all duration-300 border ${
              activePhase === phase.id
                ? 'bg-gradient-to-r from-blue-900/40 to-cyan-900/40 border-blue-500/50'
                : 'bg-black/60 border-neutral-800/50 hover:border-blue-500/30 hover:bg-blue-500/10'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center space-x-2 mb-2">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-r ${phase.color}`}>
                {phase.icon}
              </div>
              {getStatusIcon(phase.status)}
            </div>
            <h3 className="font-semibold text-white mb-1 text-xs font-[family-name:var(--font-unbounded)]">
              {phase.title}
            </h3>
            <p className="text-xs text-neutral-400 font-[family-name:var(--font-spline-sans-mono)]">
              {phase.subtitle}
            </p>
          </motion.button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {phases.map((phase) => (
          activePhase === phase.id && (
            <motion.div
              key={`${phase.id}-${animationKey}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="bg-black/80 border-neutral-800/50">
                <CardHeader>
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-r ${phase.color} text-black`}>
                      <div className="w-6 h-6">
                        {phase.icon}
                      </div>
                    </div>
                    <div>
                      <CardTitle className="text-xl md:text-2xl text-white font-[family-name:var(--font-unbounded)]">
                        {phase.title}
                      </CardTitle>
                      <CardDescription className="text-blue-400 font-semibold text-sm font-[family-name:var(--font-spline-sans-mono)]">
                        {phase.subtitle}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  <p className="text-base text-neutral-300 leading-relaxed font-[family-name:var(--font-spline-sans-mono)]">
                    {phase.description}
                  </p>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-base font-semibold text-white mb-3 font-[family-name:var(--font-unbounded)]">
                        Key Steps
                      </h4>
                      <ul className="space-y-2">
                        {phase.details.slice(0, 3).map((detail, index) => (
                          <motion.li
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="flex items-start space-x-2"
                          >
                            <div className="w-5 h-5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0 mt-0.5">
                              <span className="text-xs font-bold text-black">{index + 1}</span>
                            </div>
                            <span className="text-neutral-300 text-xs leading-relaxed font-[family-name:var(--font-spline-sans-mono)]">
                              {detail}
                            </span>
                          </motion.li>
                        ))}
                      </ul>
                    </div>

                    <div className="bg-neutral-900/50 rounded-xl p-4 border border-neutral-800/50">
                      <h4 className="text-base font-semibold text-white mb-3 flex items-center font-[family-name:var(--font-unbounded)]">
                        <Shield className="w-4 h-4 mr-2 text-blue-400" />
                        Security
                      </h4>
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <CheckCircle2 className="w-3 h-3 text-blue-400" />
                          <span className="text-xs text-neutral-300 font-[family-name:var(--font-spline-sans-mono)]">Mathematical precision</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <CheckCircle2 className="w-3 h-3 text-blue-400" />
                          <span className="text-xs text-neutral-300 font-[family-name:var(--font-spline-sans-mono)]">Invariant preservation</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )
        ))}
      </AnimatePresence>

      <div className="flex items-center justify-center space-x-3">
        <button
          onClick={() => {
            const currentIndex = phases.findIndex(p => p.id === activePhase);
            if (currentIndex > 0) {
              handlePhaseClick(phases[currentIndex - 1].id);
            }
          }}
          disabled={phases.findIndex(p => p.id === activePhase) === 0}
          className="flex items-center space-x-2 px-4 py-2 bg-black/60 hover:bg-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg border border-blue-500/30 transition-all duration-300 text-sm font-[family-name:var(--font-unbounded)]"
        >
          <ArrowRight className="w-3 h-3 rotate-180" />
          <span className="text-white">Previous</span>
        </button>
        
        <button
          onClick={() => {
            const currentIndex = phases.findIndex(p => p.id === activePhase);
            if (currentIndex < phases.length - 1) {
              handlePhaseClick(phases[currentIndex + 1].id);
            }
          }}
          disabled={phases.findIndex(p => p.id === activePhase) === phases.length - 1}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-all duration-300 text-sm font-[family-name:var(--font-unbounded)]"
        >
          <span className="text-white">Next</span>
          <ArrowRight className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
};

export default FusionTimeline; 