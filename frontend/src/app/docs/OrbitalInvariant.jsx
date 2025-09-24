import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

function OrbitalInvariant() {
  return (
    <div>
      <BlockMath math="\sum_{i=1}^{n} (r - x_i)^2 = r^2" />
    </div>
  );
}

export default OrbitalInvariant;
