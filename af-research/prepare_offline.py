#!/usr/bin/env python3
"""Reproduce computer-side preparation and stop before any formal training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from research_paths import add_firmware_arguments, resolve_source


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--tools-only', action='store_true',
                        help='Run synthetic fixture/unit tests and readiness reporting without any firmware')
    parser.add_argument('--archive', type=Path, help='Optional model ZIP/TSV cross-check')
    parser.add_argument('--no-archive', action='store_true')
    parser.add_argument('--photo', type=Path, help='Optional local photograph for explicitly unverified smoke tests')
    parser.add_argument('--step-timeout', type=int, default=300, help='Seconds per subprocess (default 300)')
    args = parser.parse_args()
    if args.tools_only and (args.photo or args.archive or args.no_archive):
        parser.error('--photo/--archive/--no-archive require the firmware workflow')
    if args.archive and args.no_archive:
        parser.error('--archive and --no-archive are mutually exclusive')
    if args.step_timeout <= 0:
        parser.error('--step-timeout must be positive')
    source = resolve_source(args)
    # Deliberately excludes probe_first_conv.py: its older toy example has an
    # artificial optimizer step. Neither workflow here calls an optimizer.
    fixture_commands = [['verify_portability.py'], ['proposal_hypothesis.py', '--self-check'],
                        ['verify_pretraining_tools.py']]
    skipped = []
    if args.tools_only:
        commands = fixture_commands + [['pretraining_readiness.py', '--tools-only']]
    else:
        source_args = ['--source', str(source)]
        extraction = ['extract.py', *source_args]
        if args.archive:
            extraction += ['--archive', str(args.archive.resolve())]
        if args.no_archive:
            extraction += ['--no-archive']
        commands = [extraction, ['decode_parameters.py', *source_args],
                    ['verify_parameters.py', '--skip-first-conv-checkpoint', *source_args],
                    ['ops_investigation.py', *source_args], ['packing_investigation.py']]
        runtime_files = ['03_program.bin', '39_hm_c_prog.bin', '40_hm_d_prog.bin',
                         '41_hm_c_ddr.bin', '42_hm_d_ddr.bin']
        missing_runtime = [name for name in runtime_files if not (source / name).is_file()]
        if missing_runtime:
            skipped.append({'command': 'runtime_investigation.py', 'reason': 'Optional runtime components unavailable',
                            'missing_files': missing_runtime})
        else:
            commands.append(['runtime_investigation.py', *source_args])
        commands += [['candidate_network.py', '--compare-layouts', *source_args],
                     ['verify_graph_hypotheses.py', *source_args], *fixture_commands,
                     ['pretraining_readiness.py', *source_args]]
        if args.photo:
            for model in ['43_hm_d_nw', '44_hm_d_nw_sng']:
                commands.append(['photo_probe.py', str(args.photo.resolve()), '--model', model,
                                 '--output', str(base / 'artifacts/photo-probe' / model),
                                 '--compare-layouts', *source_args])
    report = {'status': 'running', 'mode': 'tools_only' if args.tools_only else 'model_research',
              'formal_training_started': False, 'ready_for_formal_training': False,
              'steps': [], 'skipped_steps': skipped}
    artifacts = base / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    logs = artifacts / 'preparation-logs'
    logs.mkdir(exist_ok=True)
    report_path = artifacts / 'computer-preparation.json'

    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    save_report()  # A failed rerun must not leave an earlier success report.
    for index, command in enumerate(commands):
        print('Checking', command[0], flush=True)
        try:
            result = subprocess.run([sys.executable, *command], cwd=base, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.step_timeout)
            output, exit_code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ''
            if isinstance(output, bytes):
                output = output.decode('utf-8', errors='replace')
            output += '\nStep exceeded timeout of %d seconds.\n' % args.step_timeout
            exit_code = 124
        logfile = logs / ('%02d-%s.log' % (index, Path(command[0]).stem))
        logfile.write_text(output)
        report['steps'].append({'command': command, 'exit_code': exit_code,
                                'log': str(logfile.relative_to(base)), 'log_sha256': sha(logfile)})
        if exit_code:
            report['status'] = 'failed'
            save_report()
            print(output)
            raise SystemExit(exit_code)
    readiness = json.loads((artifacts / 'pretraining-readiness.json').read_text())
    report.update(status='tools_only_checks_complete' if args.tools_only else 'computer_side_preparation_complete',
                  ready_for_formal_training=readiness['ready_for_formal_training'],
                  outstanding=readiness['outstanding'],
                  source_code_sha256={p.name: sha(p) for p in sorted(base.glob('*.py'))},
                  scope='Offline tools and hypothesis experiments only. Original-DSP parity and physical AF remain unverified.')
    save_report()
    print('Computer-side preparation complete. Formal training started: false.')
    print('Mode:', report['mode'], '; optional steps skipped:', len(skipped))
    print('Ready for formal training:', readiness['ready_for_formal_training'])
    print('Outstanding:', ', '.join(readiness['outstanding']))


if __name__ == '__main__':
    main()
