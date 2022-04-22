/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source: keras/contraints.py */
import * as tfc from '@tensorflow/tfjs-core';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { epsilon } from './backend/common';
import { deserializeKerasObject, serializeKerasObject } from './utils/generic_utils';
/**
 * Helper function used by many of the Constraints to find the L2Norms.
 */
function calcL2Norms(w, axis) {
    return tidy(() => tfc.sqrt(tfc.sum(tfc.mul(w, w), axis, true)));
}
/**
 * Base class for functions that impose constraints on weight values
 *
 * @doc {
 *   heading: 'Constraints',
 *   subheading: 'Classes',
 *   namespace: 'constraints'
 * }
 */
export class Constraint extends serialization.Serializable {
    getConfig() {
        return {};
    }
}
export class MaxNorm extends Constraint {
    constructor(args) {
        super();
        this.defaultMaxValue = 2;
        this.defaultAxis = 0;
        this.maxValue =
            args.maxValue != null ? args.maxValue : this.defaultMaxValue;
        this.axis = args.axis != null ? args.axis : this.defaultAxis;
    }
    apply(w) {
        return tidy(() => {
            const norms = calcL2Norms(w, this.axis);
            const desired = tfc.clipByValue(norms, 0, this.maxValue);
            return tfc.mul(w, tfc.div(desired, tfc.add(epsilon(), norms)));
        });
    }
    getConfig() {
        return { maxValue: this.maxValue, axis: this.axis };
    }
}
/** @nocollapse */
MaxNorm.className = 'MaxNorm';
serialization.registerClass(MaxNorm);
export class UnitNorm extends Constraint {
    constructor(args) {
        super();
        this.defaultAxis = 0;
        this.axis = args.axis != null ? args.axis : this.defaultAxis;
    }
    apply(w) {
        return tidy(() => tfc.div(w, tfc.add(epsilon(), calcL2Norms(w, this.axis))));
    }
    getConfig() {
        return { axis: this.axis };
    }
}
/** @nocollapse */
UnitNorm.className = 'UnitNorm';
serialization.registerClass(UnitNorm);
export class NonNeg extends Constraint {
    apply(w) {
        return tfc.relu(w);
    }
}
/** @nocollapse */
NonNeg.className = 'NonNeg';
serialization.registerClass(NonNeg);
export class MinMaxNorm extends Constraint {
    constructor(args) {
        super();
        this.defaultMinValue = 0.0;
        this.defaultMaxValue = 1.0;
        this.defaultRate = 1.0;
        this.defaultAxis = 0;
        this.minValue =
            args.minValue != null ? args.minValue : this.defaultMinValue;
        this.maxValue =
            args.maxValue != null ? args.maxValue : this.defaultMaxValue;
        this.rate = args.rate != null ? args.rate : this.defaultRate;
        this.axis = args.axis != null ? args.axis : this.defaultAxis;
    }
    apply(w) {
        return tidy(() => {
            const norms = calcL2Norms(w, this.axis);
            const desired = tfc.add(tfc.mul(this.rate, tfc.clipByValue(norms, this.minValue, this.maxValue)), tfc.mul(1.0 - this.rate, norms));
            return tfc.mul(w, tfc.div(desired, tfc.add(epsilon(), norms)));
        });
    }
    getConfig() {
        return {
            minValue: this.minValue,
            maxValue: this.maxValue,
            rate: this.rate,
            axis: this.axis
        };
    }
}
/** @nocollapse */
MinMaxNorm.className = 'MinMaxNorm';
serialization.registerClass(MinMaxNorm);
// Maps the JavaScript-like identifier keys to the corresponding registry
// symbols.
export const CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP = {
    'maxNorm': 'MaxNorm',
    'minMaxNorm': 'MinMaxNorm',
    'nonNeg': 'NonNeg',
    'unitNorm': 'UnitNorm'
};
export function serializeConstraint(constraint) {
    return serializeKerasObject(constraint);
}
export function deserializeConstraint(config, customObjects = {}) {
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'constraint');
}
export function getConstraint(identifier) {
    if (identifier == null) {
        return null;
    }
    if (typeof identifier === 'string') {
        const className = identifier in CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
            CONSTRAINT_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
            identifier;
        const config = { className, config: {} };
        return deserializeConstraint(config);
    }
    else if (identifier instanceof Constraint) {
        return identifier;
    }
    else {
        return deserializeConstraint(identifier);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uc3RyYWludHMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvY29uc3RyYWludHMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCwwQ0FBMEM7QUFFMUMsT0FBTyxLQUFLLEdBQUcsTUFBTSx1QkFBdUIsQ0FBQztBQUM3QyxPQUFPLEVBQUMsYUFBYSxFQUFVLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ2xFLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUN6QyxPQUFPLEVBQUMsc0JBQXNCLEVBQUUsb0JBQW9CLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUVuRjs7R0FFRztBQUNILFNBQVMsV0FBVyxDQUFDLENBQVMsRUFBRSxJQUFZO0lBQzFDLE9BQU8sSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ2xFLENBQUM7QUFFRDs7Ozs7Ozs7R0FRRztBQUNILE1BQU0sT0FBZ0IsVUFBVyxTQUFRLGFBQWEsQ0FBQyxZQUFZO0lBR2pFLFNBQVM7UUFDUCxPQUFPLEVBQUUsQ0FBQztJQUNaLENBQUM7Q0FDRjtBQXdCRCxNQUFNLE9BQU8sT0FBUSxTQUFRLFVBQVU7SUFRckMsWUFBWSxJQUFpQjtRQUMzQixLQUFLLEVBQUUsQ0FBQztRQUpPLG9CQUFlLEdBQUcsQ0FBQyxDQUFDO1FBQ3BCLGdCQUFXLEdBQUcsQ0FBQyxDQUFDO1FBSS9CLElBQUksQ0FBQyxRQUFRO1lBQ1QsSUFBSSxDQUFDLFFBQVEsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FB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QUFVLEVBQUU7UUFDM0MsT0FBTyxVQUFVLENBQUM7S0FDbkI7U0FBTTtRQUNMLE9BQU8scUJBQXFCLENBQUMsVUFBVSxDQUFDLENBQUM7S0FDMUM7QUFDSCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyogT3JpZ2luYWwgc291cmNlOiBrZXJhcy9jb250cmFpbnRzLnB5ICovXG5cbmltcG9ydCAqIGFzIHRmYyBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtzZXJpYWxpemF0aW9uLCBUZW5zb3IsIHRpZHl9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2Vwc2lsb259IGZyb20gJy4vYmFja2VuZC9jb21tb24nO1xuaW1wb3J0IHtkZXNlcmlhbGl6ZUtlcmFzT2JqZWN0LCBzZXJpYWxpemVLZXJhc09iamVjdH0gZnJvbSAnLi91dGlscy9nZW5lcmljX3V0aWxzJztcblxuLyoqXG4gKiBIZWxwZXIgZnVuY3Rpb24gdXNlZCBieSBtYW55IG9mIHRoZSBDb25zdHJhaW50cyB0byBmaW5kIHRoZSBMMk5vcm1zLlxuICovXG5mdW5jdGlvbiBjYWxjTDJOb3Jtcyh3OiBUZW5zb3IsIGF4aXM6IG51bWJlcik6IFRlbnNvciB7XG4gIHJldHVybiB0aWR5KCgpID0+IHRmYy5zcXJ0KHRmYy5zdW0odGZjLm11bCh3LCB3KSwgYXhpcywgdHJ1ZSkpKTtcbn1cblxuLyoqXG4gKiBCYXNlIGNsYXNzIGZvciBmdW5jdGlvbnMgdGhhdCBpbXBvc2UgY29uc3RyYWludHMgb24gd2VpZ2h0IHZhbHVlc1xuICpcbiAqIEBkb2Mge1xuICogICBoZWFkaW5nOiAnQ29uc3RyYWludHMnLFxuICogICBzdWJoZWFkaW5nOiAnQ2xhc3NlcycsXG4gKiAgIG5hbWVzcGFjZTogJ2NvbnN0cmFpbnRzJ1xuICogfVxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgQ29uc3RyYWludCBleHRlbmRzIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphYmxlIHtcbiAgLyogUG9ydGluZyBub3RlOiB3YXMgX19jYWxsX18sIGFwcGx5IGNob3NlbiB0byBtYXRjaCBvdGhlciBzaW1pbGFyIGNob2ljZXMgKi9cbiAgYWJzdHJhY3QgYXBwbHkodzogVGVuc29yKTogVGVuc29yO1xuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICByZXR1cm4ge307XG4gIH1cbn1cblxuZXhwb3J0IGludGVyZmFjZSBNYXhOb3JtQXJncyB7XG4gIC8qKlxuICAgKiBNYXhpbXVtIG5vcm0gZm9yIGluY29taW5nIHdlaWdodHNcbiAgICovXG4gIG1heFZhbHVlPzogbnVtYmVyO1xuICAvKipcbiAgICogQXhpcyBhbG9uZyB3aGljaCB0byBjYWxjdWxhdGUgbm9ybXMuXG4gICAqXG4gICAqICBGb3IgaW5zdGFuY2UsIGluIGEgYERlbnNlYCBsYXllciB0aGUgd2VpZ2h0IG1hdHJpeFxuICAgKiAgaGFzIHNoYXBlIGBbaW5wdXREaW0sIG91dHB1dERpbV1gLFxuICAgKiAgc2V0IGBheGlzYCB0byBgMGAgdG8gY29uc3RyYWluIGVhY2ggd2VpZ2h0IHZlY3RvclxuICAgKiAgb2YgbGVuZ3RoIGBbaW5wdXREaW0sXWAuXG4gICAqICBJbiBhIGBDb252MkRgIGxheWVyIHdpdGggYGRhdGFGb3JtYXQ9XCJjaGFubmVsc19sYXN0XCJgLFxuICAgKiAgdGhlIHdlaWdodCB0ZW5zb3IgaGFzIHNoYXBlXG4gICAqICBgW3Jvd3MsIGNvbHMsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXWAsXG4gICAqICBzZXQgYGF4aXNgIHRvIGBbMCwgMSwgMl1gXG4gICAqICB0byBjb25zdHJhaW4gdGhlIHdlaWdodHMgb2YgZWFjaCBmaWx0ZXIgdGVuc29yIG9mIHNpemVcbiAgICogIGBbcm93cywgY29scywgaW5wdXREZXB0aF1gLlxuICAgKi9cbiAgYXhpcz86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIE1heE5vcm0gZXh0ZW5kcyBDb25zdHJhaW50IHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyByZWFkb25seSBjbGFzc05hbWUgPSAnTWF4Tm9ybSc7XG4gIHByaXZhdGUgbWF4VmFsdWU6IG51bWJlcjtcbiAgcHJpdmF0ZSBheGlzOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgZGVmYXVsdE1heFZhbHVlID0gMjtcbiAgcHJpdmF0ZSByZWFkb25seSBkZWZhdWx0QXhpcyA9IDA7XG5cbiAgY29uc3RydWN0b3IoYXJnczogTWF4Tm9ybUFyZ3MpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMubWF4VmFsdWUgPVxuICAgICAgICBhcmdzLm1heFZhbHVlICE9IG51bGwgPyBhcmdzLm1heFZhbHVlIDogdGhpcy5kZWZhdWx0TWF4VmFsdWU7XG4gICAgdGhpcy5heGlzID0gYXJncy5heGlzICE9IG51bGwgPyBhcmdzLmF4aXMgOiB0aGlzLmRlZmF1bHRBeGlzO1xuICB9XG5cbiAgYXBwbHkodzogVGVuc29yKTogVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBub3JtcyA9IGNhbGNMMk5vcm1zKHcsIHRoaXMuYXhpcyk7XG4gICAgICBjb25zdCBkZXNpcmVkID0gdGZjLmNsaXBCeVZhbHVlKG5vcm1zLCAwLCB0aGlzLm1heFZhbHVlKTtcbiAgICAgIHJldHVybiB0ZmMubXVsKHcsIHRmYy5kaXYoZGVzaXJlZCwgdGZjLmFkZChlcHNpbG9uKCksIG5vcm1zKSkpO1xuICAgIH0pO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHttYXhWYWx1ZTogdGhpcy5tYXhWYWx1ZSwgYXhpczogdGhpcy5heGlzfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1heE5vcm0pO1xuXG5leHBvcnQgaW50ZXJmYWNlIFVuaXROb3JtQXJncyB7XG4gIC8qKlxuICAgKiBBeGlzIGFsb25nIHdoaWNoIHRvIGNhbGN1bGF0ZSBub3Jtcy5cbiAgICpcbiAgICogRm9yIGluc3RhbmNlLCBpbiBhIGBEZW5zZWAgbGF5ZXIgdGhlIHdlaWdodCBtYXRyaXhcbiAgICogaGFzIHNoYXBlIGBbaW5wdXREaW0sIG91dHB1dERpbV1gLFxuICAgKiBzZXQgYGF4aXNgIHRvIGAwYCB0byBjb25zdHJhaW4gZWFjaCB3ZWlnaHQgdmVjdG9yXG4gICAqIG9mIGxlbmd0aCBgW2lucHV0RGltLF1gLlxuICAgKiBJbiBhIGBDb252MkRgIGxheWVyIHdpdGggYGRhdGFGb3JtYXQ9XCJjaGFubmVsc19sYXN0XCJgLFxuICAgKiB0aGUgd2VpZ2h0IHRlbnNvciBoYXMgc2hhcGVcbiAgICogW3Jvd3MsIGNvbHMsIGlucHV0RGVwdGgsIG91dHB1dERlcHRoXWAsXG4gICAqIHNldCBgYXhpc2AgdG8gYFswLCAxLCAyXWBcbiAgICogdG8gY29uc3RyYWluIHRoZSB3ZWlnaHRzIG9mIGVhY2ggZmlsdGVyIHRlbnNvciBvZiBzaXplXG4gICAqIGBbcm93cywgY29scywgaW5wdXREZXB0aF1gLlxuICAgKi9cbiAgYXhpcz86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIFVuaXROb3JtIGV4dGVuZHMgQ29uc3RyYWludCB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgcmVhZG9ubHkgY2xhc3NOYW1lID0gJ1VuaXROb3JtJztcbiAgcHJpdmF0ZSBheGlzOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgZGVmYXVsdEF4aXMgPSAwO1xuICBjb25zdHJ1Y3RvcihhcmdzOiBVbml0Tm9ybUFyZ3MpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMuYXhpcyA9IGFyZ3MuYXhpcyAhPSBudWxsID8gYXJncy5heGlzIDogdGhpcy5kZWZhdWx0QXhpcztcbiAgfVxuXG4gIGFwcGx5KHc6IFRlbnNvcik6IFRlbnNvciB7XG4gICAgcmV0dXJuIHRpZHkoXG4gICAgICAgICgpID0+IHRmYy5kaXYodywgdGZjLmFkZChlcHNpbG9uKCksIGNhbGNMMk5vcm1zKHcsIHRoaXMuYXhpcykpKSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICByZXR1cm4ge2F4aXM6IHRoaXMuYXhpc307XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhVbml0Tm9ybSk7XG5cbmV4cG9ydCBjbGFzcyBOb25OZWcgZXh0ZW5kcyBDb25zdHJhaW50IHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyByZWFkb25seSBjbGFzc05hbWUgPSAnTm9uTmVnJztcblxuICBhcHBseSh3OiBUZW5zb3IpOiBUZW5zb3Ige1xuICAgIHJldHVybiB0ZmMucmVsdSh3KTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE5vbk5lZyk7XG5cbmV4cG9ydCBpbnRlcmZhY2UgTWluTWF4Tm9ybUFyZ3Mge1xuICAvKipcbiAgICogTWluaW11bSBub3JtIGZvciBpbmNvbWluZyB3ZWlnaHRzXG4gICAqL1xuICBtaW5WYWx1ZT86IG51bWJlcjtcbiAgLyoqXG4gICAqIE1heGltdW0gbm9ybSBmb3IgaW5jb21pbmcgd2VpZ2h0c1xuICAgKi9cbiAgbWF4VmFsdWU/OiBudW1iZXI7XG4gIC8qKlxuICAgKiBBeGlzIGFsb25nIHdoaWNoIHRvIGNhbGN1bGF0ZSBub3Jtcy5cbiAgICogRm9yIGluc3RhbmNlLCBpbiBhIGBEZW5zZWAgbGF5ZXIgdGhlIHdlaWdodCBtYXRyaXhcbiAgICogaGFzIHNoYXBlIGBbaW5wdXREaW0sIG91dHB1dERpbV1gLFxuICAgKiBzZXQgYGF4aXNgIHRvIGAwYCB0byBjb25zdHJhaW4gZWFjaCB3ZWlnaHQgdmVjdG9yXG4gICAqIG9mIGxlbmd0aCBgW2lucHV0RGltLF1gLlxuICAgKiBJbiBhIGBDb252MkRgIGxheWVyIHdpdGggYGRhdGFGb3JtYXQ9XCJjaGFubmVsc19sYXN0XCJgLFxuICAgKiB0aGUgd2VpZ2h0IHRlbnNvciBoYXMgc2hhcGVcbiAgICogYFtyb3dzLCBjb2xzLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF1gLFxuICAgKiBzZXQgYGF4aXNgIHRvIGBbMCwgMSwgMl1gXG4gICAqIHRvIGNvbnN0cmFpbiB0aGUgd2VpZ2h0cyBvZiBlYWNoIGZpbHRlciB0ZW5zb3Igb2Ygc2l6ZVxuICAgKiBgW3Jvd3MsIGNvbHMsIGlucHV0RGVwdGhdYC5cbiAgICovXG4gIGF4aXM/OiBudW1iZXI7XG4gIC8qKlxuICAgKiBSYXRlIGZvciBlbmZvcmNpbmcgdGhlIGNvbnN0cmFpbnQ6IHdlaWdodHMgd2lsbCBiZSByZXNjYWxlZCB0byB5aWVsZDpcbiAgICogYCgxIC0gcmF0ZSkgKiBub3JtICsgcmF0ZSAqIG5vcm0uY2xpcChtaW5WYWx1ZSwgbWF4VmFsdWUpYC5cbiAgICogRWZmZWN0aXZlbHksIHRoaXMgbWVhbnMgdGhhdCByYXRlPTEuMCBzdGFuZHMgZm9yIHN0cmljdFxuICAgKiBlbmZvcmNlbWVudCBvZiB0aGUgY29uc3RyYWludCwgd2hpbGUgcmF0ZTwxLjAgbWVhbnMgdGhhdFxuICAgKiB3ZWlnaHRzIHdpbGwgYmUgcmVzY2FsZWQgYXQgZWFjaCBzdGVwIHRvIHNsb3dseSBtb3ZlXG4gICAqIHRvd2FyZHMgYSB2YWx1ZSBpbnNpZGUgdGhlIGRlc2lyZWQgaW50ZXJ2YWwuXG4gICAqL1xuICByYXRlPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgTWluTWF4Tm9ybSBleHRlbmRzIENvbnN0cmFpbnQge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIHJlYWRvbmx5IGNsYXNzTmFtZSA9ICdNaW5NYXhOb3JtJztcbiAgcHJpdmF0ZSBtaW5WYWx1ZTogbnVtYmVyO1xuICBwcml2YXRlIG1heFZhbHVlOiBudW1iZXI7XG4gIHByaXZhdGUgcmF0ZTogbnVtYmVyO1xuICBwcml2YXRlIGF4aXM6IG51bWJlcjtcbiAgcHJpdmF0ZSByZWFkb25seSBkZWZhdWx0TWluVmFsdWUgPSAwLjA7XG4gIHByaXZhdGUgcmVhZG9ubHkgZGVmYXVsdE1heFZhbHVlID0gMS4wO1xuICBwcml2YXRlIHJlYWRvbmx5IGRlZmF1bHRSYXRlID0gMS4wO1xuICBwcml2YXRlIHJlYWRvbmx5IGRlZmF1bHRBeGlzID0gMDtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBNaW5NYXhOb3JtQXJncykge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5taW5WYWx1ZSA9XG4gICAgICAgIGFyZ3MubWluVmFsdWUgIT0gbnVsbCA/IGFyZ3MubWluVmFsdWUgOiB0aGlzLmRlZmF1bHRNaW5WYWx1ZTtcbiAgICB0aGlzLm1heFZhbHVlID1cbiAgICAgICAgYXJncy5tYXhWYWx1ZSAhPSBudWxsID8gYXJncy5tYXhWYWx1ZSA6IHRoaXMuZGVmYXVsdE1heFZhbHVlO1xuICAgIHRoaXMucmF0ZSA9IGFyZ3MucmF0ZSAhPSBudWxsID8gYXJncy5yYXRlIDogdGhpcy5kZWZhdWx0UmF0ZTtcbiAgICB0aGlzLmF4aXMgPSBhcmdzLmF4aXMgIT0gbnVsbCA/IGFyZ3MuYXhpcyA6IHRoaXMuZGVmYXVsdEF4aXM7XG4gIH1cblxuICBhcHBseSh3OiBUZW5zb3IpOiBUZW5zb3Ige1xuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IG5vcm1zID0gY2FsY0wyTm9ybXModywgdGhpcy5heGlzKTtcbiAgICAgIGNvbnN0IGRlc2lyZWQgPSB0ZmMuYWRkKFxuICAgICAgICAgIHRmYy5tdWwoXG4gICAgICAgICAgICAgIHRoaXMucmF0ZSwgdGZjLmNsaXBCeVZhbHVlKG5vcm1zLCB0aGlzLm1pblZhbHVlLCB0aGlzLm1heFZhbHVlKSksXG4gICAgICAgICAgdGZjLm11bCgxLjAgLSB0aGlzLnJhdGUsIG5vcm1zKSk7XG4gICAgICByZXR1cm4gdGZjLm11bCh3LCB0ZmMuZGl2KGRlc2lyZWQsIHRmYy5hZGQoZXBzaWxvbigpLCBub3JtcykpKTtcbiAgICB9KTtcbiAgfVxuXG4gIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7XG4gICAgICBtaW5WYWx1ZTogdGhpcy5taW5WYWx1ZSxcbiAgICAgIG1heFZhbHVlOiB0aGlzLm1heFZhbHVlLFxuICAgICAgcmF0ZTogdGhpcy5yYXRlLFxuICAgICAgYXhpczogdGhpcy5heGlzXG4gICAgfTtcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKE1pbk1heE5vcm0pO1xuXG4vKiogQGRvY2lubGluZSAqL1xuZXhwb3J0IHR5cGUgQ29uc3RyYWludElkZW50aWZpZXIgPVxuICAgICdtYXhOb3JtJ3wnbWluTWF4Tm9ybSd8J25vbk5lZyd8J3VuaXROb3JtJ3xzdHJpbmc7XG5cbi8vIE1hcHMgdGhlIEphdmFTY3JpcHQtbGlrZSBpZGVudGlmaWVyIGtleXMgdG8gdGhlIGNvcnJlc3BvbmRpbmcgcmVnaXN0cnlcbi8vIHN5bWJvbHMuXG5leHBvcnQgY29uc3QgQ09OU1RSQUlOVF9JREVOVElGSUVSX1JFR0lTVFJZX1NZTUJPTF9NQVA6XG4gICAge1tpZGVudGlmaWVyIGluIENvbnN0cmFpbnRJZGVudGlmaWVyXTogc3RyaW5nfSA9IHtcbiAgICAgICdtYXhOb3JtJzogJ01heE5vcm0nLFxuICAgICAgJ21pbk1heE5vcm0nOiAnTWluTWF4Tm9ybScsXG4gICAgICAnbm9uTmVnJzogJ05vbk5lZycsXG4gICAgICAndW5pdE5vcm0nOiAnVW5pdE5vcm0nXG4gICAgfTtcblxuZXhwb3J0IGZ1bmN0aW9uIHNlcmlhbGl6ZUNvbnN0cmFpbnQoY29uc3RyYWludDogQ29uc3RyYWludCk6XG4gICAgc2VyaWFsaXphdGlvbi5Db25maWdEaWN0VmFsdWUge1xuICByZXR1cm4gc2VyaWFsaXplS2VyYXNPYmplY3QoY29uc3RyYWludCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZXNlcmlhbGl6ZUNvbnN0cmFpbnQoXG4gICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QsXG4gICAgY3VzdG9tT2JqZWN0czogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge30pOiBDb25zdHJhaW50IHtcbiAgcmV0dXJuIGRlc2VyaWFsaXplS2VyYXNPYmplY3QoXG4gICAgICBjb25maWcsIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphdGlvbk1hcC5nZXRNYXAoKS5jbGFzc05hbWVNYXAsXG4gICAgICBjdXN0b21PYmplY3RzLCAnY29uc3RyYWludCcpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Q29uc3RyYWludChpZGVudGlmaWVyOiBDb25zdHJhaW50SWRlbnRpZmllcnxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdHxDb25zdHJhaW50KTogQ29uc3RyYWludCB7XG4gIGlmIChpZGVudGlmaWVyID09IG51bGwpIHtcbiAgICByZXR1cm4gbnVsbDtcbiAgfVxuICBpZiAodHlwZW9mIGlkZW50aWZpZXIgPT09ICdzdHJpbmcnKSB7XG4gICAgY29uc3QgY2xhc3NOYW1lID0gaWRlbnRpZmllciBpbiBDT05TVFJBSU5UX0lERU5USUZJRVJfUkVHSVNUUllfU1lNQk9MX01BUCA/XG4gICAgICAgIENPTlNUUkFJTlRfSURFTlRJRklFUl9SRUdJU1RSWV9TWU1CT0xfTUFQW2lkZW50aWZpZXJdIDpcbiAgICAgICAgaWRlbnRpZmllcjtcbiAgICBjb25zdCBjb25maWcgPSB7Y2xhc3NOYW1lLCBjb25maWc6IHt9fTtcbiAgICByZXR1cm4gZGVzZXJpYWxpemVDb25zdHJhaW50KGNvbmZpZyk7XG4gIH0gZWxzZSBpZiAoaWRlbnRpZmllciBpbnN0YW5jZW9mIENvbnN0cmFpbnQpIHtcbiAgICByZXR1cm4gaWRlbnRpZmllcjtcbiAgfSBlbHNlIHtcbiAgICByZXR1cm4gZGVzZXJpYWxpemVDb25zdHJhaW50KGlkZW50aWZpZXIpO1xuICB9XG59XG4iXX0=