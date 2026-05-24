import { motion } from 'framer-motion';

export function LoadingState() {
  return (
    <div className="flex items-center justify-center space-x-1 h-6">
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.div
          key={i}
          className="w-1 bg-white rounded-full"
          initial={{ height: 8, opacity: 0.5 }}
          animate={{ 
            height: [8, 24, 8], 
            opacity: [0.5, 1, 0.5] 
          }}
          transition={{
            duration: 1,
            repeat: Infinity,
            delay: i * 0.1,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  );
}
