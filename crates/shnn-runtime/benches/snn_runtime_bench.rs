use criterion::{criterion_group, criterion_main, BenchmarkId, BatchSize, Criterion, Throughput};
use shnn_runtime::{network::NetworkBuilder, simulation::run_fixed_step, NeuronId};

fn build_network(neurons: u32, fully_connected: bool, weight: f32) -> shnn_runtime::network::SNNNetwork {
    let mut builder = NetworkBuilder::new().add_neurons(0, neurons);
    if fully_connected {
        // Keep this small in benches to avoid explosive edges
        builder = builder.fully_connected(weight);
    } else if neurons >= 2 {
        // Simple chain
        for i in 0..(neurons - 1) {
            builder = builder.add_synapse_simple(NeuronId::new(i), NeuronId::new(i + 1), weight);
        }
    }
    builder.build().expect("bench network build")
}

fn bench_fixed_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("snn_runtime_fixed_step");
    // Short duration to keep benches fast in CI
    let dt_ns = 100_000; // 0.1 ms
    let duration_ns = 2_000_000; // 2 ms

    for &n in &[8u32, 16u32, 32u32] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, &_n| {
            b.iter_batched(
                || build_network(n, false, 0.2),
                |net| {
                    let _res = run_fixed_step(net, dt_ns, duration_ns, Some(1234)).unwrap();
                },
                BatchSize::SmallInput,
            );
        });

        // Keep fully-connected only for smallest case to avoid long CI times
        if n <= 16 {
            group.bench_with_input(BenchmarkId::new("fully_connected", n), &n, |b, &_n| {
                b.iter_batched(
                    || build_network(n, true, 0.1),
                    |net| {
                        let _res = run_fixed_step(net, dt_ns, duration_ns, Some(1234)).unwrap();
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_fixed_step);
criterion_main!(benches);