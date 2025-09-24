import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ArrowRight, 
  CircleDot,
  CheckCircle,
  Settings,
  Zap,
  Activity
} from 'lucide-react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

const Card = ({ children, className = "" }: CardProps) => (
  <div className={`rounded-lg border bg-card text-card-foreground shadow-sm ${className}`}>
    {children}
  </div>
);

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

const CardHeader = ({ children, className = "" }: CardHeaderProps) => (
  <div className={`flex flex-col space-y-1.5 p-6 ${className}`}>
    {children}
  </div>
);

interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
}

const CardTitle = ({ children, className = "" }: CardTitleProps) => (
  <h3 className={`text-2xl font-semibold leading-none tracking-tight ${className}`}>
    {children}
  </h3>
);

interface CardDescriptionProps {
  children: React.ReactNode;
  className?: string;
}

const CardDescription = ({ children, className = "" }: CardDescriptionProps) => (
  <p className={`text-sm text-muted-foreground ${className}`}>
    {children}
  </p>
);

interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

const CardContent = ({ children, className = "" }: CardContentProps) => (
  <div className={`p-6 pt-0 ${className}`}>
    {children}
  </div>
);

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "default" | "outline" | "gradient";
  className?: string;
}

const Button = ({ children, onClick, variant = "default", className = "", ...props }: ButtonProps) => {
  const baseClasses = "inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2";
  const variantClasses: Record<string, string> = {
    default: "bg-primary text-primary-foreground hover:bg-primary/90",
    outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
    gradient: "text-white font-semibold shadow-lg"
  };
  
  return (
    <button 
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

interface ProgressProps {
  value?: number;
  className?: string;
}

const Progress = ({ value = 0, className = "" }: ProgressProps) => (
  <div className={`relative h-2 w-full overflow-hidden rounded-full bg-neutral-800 ${className}`}>
    <div 
      className="h-full w-full flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
      style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
    />
  </div>
);

interface OrbitalVisualizationProps {
  phase: string;
  progress: number;
}

const OrbitalVisualization = ({ phase }: OrbitalVisualizationProps) => {
  const isActive = phase !== 'idle';
  
  return (
    <motion.div 
      className="relative w-full flex items-center justify-center bg-gradient-to-r from-blue-900/10 via-black to-black rounded-xl border border-blue-500/20"
      initial={{ height: 0, opacity: 0 }}
      animate={{ 
        height: isActive ? 320 : 0,
        opacity: isActive ? 1 : 0
      }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      {isActive && (
        <svg width="300" height="300" className="absolute">
          <defs>
            <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
              <stop offset="70%" stopColor="#06b6d4" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0" />
            </radialGradient>
            
            <linearGradient id="orbitalPath" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
              <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.4" />
            </linearGradient>

            <filter id="softGlow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          <g opacity="0.4">
            <circle cx="150" cy="150" r="60" fill="none" stroke="url(#orbitalPath)" strokeWidth="1" strokeDasharray="4,8" />
            <circle cx="150" cy="150" r="90" fill="none" stroke="url(#orbitalPath)" strokeWidth="1" strokeDasharray="4,8" />
            <circle cx="150" cy="150" r="120" fill="none" stroke="url(#orbitalPath)" strokeWidth="1" strokeDasharray="4,8" />
          </g>

          <motion.circle
            cx="150"
            cy="150"
            r="30"
            fill="url(#coreGlow)"
            filter="url(#softGlow)"
            animate={{
              r: [30, 35, 30],
              opacity: [0.6, 1, 0.6]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />

          {phase === 'tick-matching' && (
            <g>
              {[0, 1, 2, 3, 4, 5].map((i) => {
                const angle = (i * 60) * Math.PI / 180;
                const radius = 80;
                const x = 150 + Math.cos(angle) * radius;
                const y = 150 + Math.sin(angle) * radius;
                
                return (
                  <motion.circle
                    key={`tick-${i}`}
                    cx={x}
                    cy={y}
                    r="4"
                    fill="#3b82f6"
                    filter="url(#softGlow)"
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ 
                      scale: [0, 1, 0.8],
                      opacity: [0, 1, 0.6]
                    }}
                    transition={{
                      duration: 2,
                      delay: i * 0.2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                );
              })}
            </g>
          )}

          {phase === 'sphere-consolidation' && (
            <motion.circle
              cx="150"
              cy="150"
              r="60"
              fill="none"
              stroke="#eab308"
              strokeWidth="2"
              strokeOpacity="0.6"
              animate={{
                scale: [1, 1.2, 1],
                strokeOpacity: [0.6, 0.2, 0.6]
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
          )}

          {phase === 'orbital-settlement' && (
            <motion.circle
              cx="150"
              cy="150"
              r="30"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="3"
              strokeOpacity="0.8"
              animate={{
                r: [30, 100, 30],
                strokeOpacity: [0.8, 0, 0.8],
                strokeWidth: [3, 1, 3]
              }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
                ease: "easeOut"
              }}
            />
          )}

          {phase === 'complete' && (
            <g>
              {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
                const angle = (i * 45) * Math.PI / 180;
                const startRadius = 30;
                const endRadius = 120;
                const startX = 150 + Math.cos(angle) * startRadius;
                const startY = 150 + Math.sin(angle) * startRadius;
                const endX = 150 + Math.cos(angle) * endRadius;
                const endY = 150 + Math.sin(angle) * endRadius;
                
                return (
                  <motion.line
                    key={`burst-${i}`}
                    x1="150"
                    y1="150"
                    x2={startX}
                    y2={startY}
                    stroke="#06b6d4"
                    strokeWidth="2"
                    strokeOpacity="0.8"
                    animate={{
                      x2: endX,
                      y2: endY,
                      strokeOpacity: [0.8, 0]
                    }}
                    transition={{
                      duration: 1,
                      delay: i * 0.1,
                      ease: "easeOut"
                    }}
                  />
                );
              })}
            </g>
          )}

          <circle cx="150" cy="150" r="45" fill="none" stroke="#333" strokeWidth="2" />
          <circle 
            cx="150" 
            cy="150" 
            r="45" 
            fill="none" 
            stroke="#3b82f6" 
            strokeWidth="2" 
            strokeDasharray="283"
            strokeDashoffset="0"
            transform="rotate(-90 150 150)"
          />
        </svg>
      )}
    </motion.div>
  );
};

const InteractiveDemo = () => {
  const [demoState, setDemoState] = useState({
    phase: 'idle',
    progress: 0,
    isPlaying: false,
    stepIndex: 0
  });

  const [swapDetails] = useState({
    fromToken: 'USDC',
    toToken: 'USDT',
    fromChain: '',
    toChain: '',
    amount: '10,000',
    estimatedOutput: '9,998'
  });

  const demoSteps = useMemo(() => [
    {
      phase: 'tick-matching',
      title: 'Identifying Optimal Liquidity Ticks',
      duration: 4000,
      events: [
        'Analyzing stablecoin price deviations from $1 peg',
        'Computing geodesic distances from equal-price point',
        'Selecting optimal tick boundaries via plane constraints',
        'Matching liquidity to tick radius and depth',
        'Activating concentrated liquidity positions'
      ]
    },
    {
      phase: 'sphere-consolidation',
      title: 'Consolidating Interior Ticks into Sphere AMM',
      duration: 3500,
      events: [
        'Interior ticks behaving as unified spherical surface',
        'Boundary ticks pinned to orthogonal subspaces',
        'Computing torus invariant for multi-dimensional trading',
        'Capital efficiency gains: 15-150x vs uniform AMM',
        'Trade routing through consolidated liquidity'
      ]
    },
    {
      phase: 'orbital-settlement',
      title: 'Executing Trade via Torus Surface Geometry',
      duration: 2500,
      events: [
        'Computing quartic trade equations in real-time',
        'Executing atomic swap via torus surface geometry',
        'Maintaining tick boundary constraints during trade',
        'No-arbitrage conditions enforced across dimensions',
        'Settlement complete with minimal slippage'
      ]
    }
  ], []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (demoState.isPlaying && demoState.phase !== 'complete') {
      interval = setInterval(() => {
        setDemoState(prev => {
          const currentStep = demoSteps.find(step => step.phase === prev.phase);
          if (!currentStep) return prev;
          
          const newProgress = prev.progress + (100 / (currentStep.duration / 100));
          
          if (newProgress >= 100) {
            const currentStepIndex = demoSteps.findIndex(step => step.phase === prev.phase);
            const nextStep = demoSteps[currentStepIndex + 1];
            
            if (nextStep) {
              return {
                ...prev,
                phase: nextStep.phase,
                progress: 0,
                stepIndex: currentStepIndex + 1
              };
            } else {
              return {
                ...prev,
                phase: 'complete',
                progress: 100,
                isPlaying: false
              };
            }
          }
          
          return { ...prev, progress: newProgress };
        });
      }, 100);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [demoState.isPlaying, demoState.phase, demoSteps]);

  const handlePlay = () => {
    if (demoState.phase === 'idle' || demoState.phase === 'complete') {
      setDemoState({
        phase: 'tick-matching',
        progress: 0,
        isPlaying: true,
        stepIndex: 0
      });
    } else {
      setDemoState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
    }
  };

  const handleReset = () => {
    setDemoState({
      phase: 'idle',
      progress: 0,
      isPlaying: false,
      stepIndex: 0
    });
  };

  const getCurrentStep = () => {
    return demoSteps.find(step => step.phase === demoState.phase) || demoSteps[0];
  };

  const getPhaseProgress = () => {
    if (demoState.phase === 'complete') return 100;
    const baseProgress = (demoState.stepIndex / demoSteps.length) * 100;
    const stepProgress = (demoState.progress / demoSteps.length);
    return baseProgress + stepProgress;
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-black/30">
      <div className="text-center space-y-4">
        <motion.h1 
          className="text-3xl md:text-4xl font-black bg-white bg-clip-text text-transparent"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Orbital Protocol
        </motion.h1>
        <motion.p 
          className="text-xl py-5 text-neutral-300 max-w-3xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Experience <span className="text-blue-400 font-semibold">concentrated liquidity in higher dimensions</span> through advanced mathematical optimization
        </motion.p>
      </div>

      <motion.div 
        className="grid md:grid-cols-2 gap-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Card className="bg-gradient-to-br from-black via-neutral-900/50 to-black border-blue-500/20">
          <CardHeader>
            <CardTitle className="flex items-center text-white">
              <Settings className="w-6 h-6 mr-3 text-blue-400" />
              Multi-Dimensional AMM
            </CardTitle>
            <CardDescription className="text-neutral-300">
              Stablecoin trading with orbital mechanics
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="p-4 bg-black/60 rounded-xl border border-blue-500/30 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-neutral-400 text-sm">From</span>
                  <span className="text-blue-400 font-medium text-sm">{swapDetails.fromChain}</span>
                </div>
                <div className="text-3xl font-bold text-white">
                  {swapDetails.amount} {swapDetails.fromToken}
                </div>
                <div className="text-xs text-neutral-500 mt-1">
                  Interior Tick Active
                </div>
              </div>

              <div className="p-4 bg-black/60 rounded-xl border border-blue-500/30 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-neutral-400 text-sm">To</span>
                  <span className="text-cyan-400 font-medium text-sm">{swapDetails.toChain}</span>
                </div>
                <div className="text-3xl font-bold text-white">
                  {swapDetails.estimatedOutput} {swapDetails.toToken}
                </div>
                <div className="text-xs text-neutral-500 mt-1">
                  Price: $0.999 
                </div>
              </div>
            </div>

            <div className="flex space-x-3 pt-4">
              <Button
                onClick={handlePlay}
                variant="gradient"
                className="flex-1 bg-gradient-to-r cursor-pointer from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 shadow-lg shadow-blue-500/20"
              >
                {demoState.isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                {demoState.phase === 'idle' ? 'Start Orbital Demo' : demoState.isPlaying ? 'Pause' : 'Resume'}
              </Button>
              <Button 
                onClick={handleReset} 
                variant="outline" 
                className="border-blue-500/50 text-blue-400 cursor-pointer hover:bg-blue-500/10"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-black via-neutral-900/50 to-black border-blue-500/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center text-white">
                  <Zap className="w-6 h-6 mr-3 text-blue-400" />
                  Orbital Mechanics
                </CardTitle>
                <CardDescription className="text-neutral-300">
                  {demoState.phase === 'idle' ? 'Ready to initialize orbital trading' : 
                   demoState.phase === 'complete' ? 'Trade executed successfully' :
                   getCurrentStep().title}
                </CardDescription>
              </div>
              {demoState.phase !== 'idle' && demoState.phase !== 'complete' && (
                <div className="text-sm text-blue-400">
                  {Math.round(getPhaseProgress())}%
                </div>
              )}
            </div>
          </CardHeader>
          
          <CardContent className="space-y-6">
            <OrbitalVisualization phase={demoState.phase} progress={demoState.progress} />

            <AnimatePresence>
              {demoState.phase !== 'idle' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-300">
                      Trade Progress
                    </span>
                    <span className="text-sm font-medium text-blue-400">
                      {Math.round(getPhaseProgress())}%
                    </span>
                  </div>
                  <Progress value={getPhaseProgress()} />
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence mode="wait">
              {demoState.phase === 'idle' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center py-8"
                >
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center text-black shadow-lg shadow-blue-500/20">
                    <CircleDot className="w-8 h-8" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">
                    Orbital AMM Ready
                  </h3>
                  <p className="text-neutral-400">
                    Initialize multi-dimensional liquidity optimization
                  </p>
                </motion.div>
              )}

              {demoState.phase !== 'idle' && demoState.phase !== 'complete' && (
                <motion.div
                  key={demoState.phase}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="bg-gradient-to-br from-neutral-900/60 to-black rounded-xl p-6 border border-blue-500/20 backdrop-blur-sm"
                >
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center text-black">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {getCurrentStep().title}
                      </h3>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    {getCurrentStep().events.map((event, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ 
                          opacity: (demoState.progress / 100) > (index / getCurrentStep().events.length) ? 1 : 0.4,
                          x: 0 
                        }}
                        className="flex items-start space-x-3"
                      >
                        <div className="mt-1 flex-shrink-0">
                          {(demoState.progress / 100) > (index / getCurrentStep().events.length) ? (
                            <CheckCircle className="w-4 h-4 text-blue-400" />
                          ) : (
                            <div className="w-4 h-4 rounded-full border border-neutral-600" />
                          )}
                        </div>
                        <span className="text-sm text-neutral-300 leading-relaxed">
                          {event}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {demoState.phase === 'complete' && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-gradient-to-br from-neutral-900/60 to-black rounded-xl p-8 border border-blue-500/40 text-center backdrop-blur-sm"
                >
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center text-black shadow-lg shadow-blue-500/20">
                    <CheckCircle className="w-8 h-8" />
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">
                    Orbital Trade Complete!
                  </h3>
                  <p className="text-neutral-300 mb-6">
                    Cross-dimensional swap executed via torus surface geometry
                  </p>
                  
                  <div className="flex items-center justify-center space-x-4 mb-6">
                    <div className="bg-black/60 rounded-lg px-4 py-3 border border-blue-500/30">
                      <span className="text-neutral-400 text-sm">Sent:</span>
                      <div className="text-white font-semibold">
                        {swapDetails.amount} {swapDetails.fromToken}
                      </div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-blue-400" />
                    <div className="bg-black/60 rounded-lg px-4 py-3 border border-blue-500/30">
                      <span className="text-neutral-400 text-sm">Received:</span>
                      <div className="text-white font-semibold">
                        {swapDetails.estimatedOutput} {swapDetails.toToken}
                      </div>
                    </div>
                  </div>
                  
        
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </Card>

      </motion.div>
    </div>
  );
};

export default InteractiveDemo;