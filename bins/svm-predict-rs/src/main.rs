use libsvm_rs::io::{format_17g, format_g, load_model, load_problem};
use libsvm_rs::predict::{predict, predict_probability};
use libsvm_rs::{SvmModel, SvmType};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process;

fn exit_with_help() -> ! {
    print!(
        "\
Usage: svm-predict [options] test_file model_file output_file
options:
-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported
-q : quiet mode (no outputs)
"
    );
    process::exit(1);
}

fn check_probability_model(model: &SvmModel) -> bool {
    match model.param.svm_type {
        SvmType::CSvc | SvmType::NuSvc => {
            !model.prob_a.is_empty() && !model.prob_b.is_empty()
        }
        SvmType::EpsilonSvr | SvmType::NuSvr => !model.prob_a.is_empty(),
        SvmType::OneClass => !model.prob_density_marks.is_empty(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut predict_prob = false;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        if !args[i].starts_with('-') {
            break;
        }
        let flag = &args[i];

        if flag == "-q" {
            quiet = true;
            i += 1;
            continue;
        }

        i += 1;
        if i >= args.len() {
            exit_with_help();
        }

        match flag.as_bytes()[1] {
            b'b' => {
                predict_prob = args[i].parse::<i32>().unwrap_or(0) != 0;
            }
            _ => {
                eprintln!("Unknown option: {}", flag);
                exit_with_help();
            }
        }
        i += 1;
    }

    // Need exactly 3 remaining args: test_file model_file output_file
    if i + 2 >= args.len() {
        exit_with_help();
    }
    let test_file = &args[i];
    let model_file = &args[i + 1];
    let output_file = &args[i + 2];

    // Quiet mode
    if quiet {
        libsvm_rs::set_quiet(true);
    }

    // Load model
    let model = load_model(Path::new(model_file)).unwrap_or_else(|e| {
        eprintln!("can't open model file {}: {}", model_file, e);
        process::exit(1);
    });

    // Check probability support
    if predict_prob {
        if !check_probability_model(&model) {
            eprintln!("Model does not support probabiliy estimates");
            process::exit(1);
        }
    } else if check_probability_model(&model) && !quiet {
        eprintln!("Model supports probability estimates, but disabled in prediction.");
    }

    // Load test data
    let problem = load_problem(Path::new(test_file)).unwrap_or_else(|e| {
        eprintln!("can't open input file {}: {}", test_file, e);
        process::exit(1);
    });

    // Open output file
    let out = File::create(output_file).unwrap_or_else(|e| {
        eprintln!("can't open output file {}: {}", output_file, e);
        process::exit(1);
    });
    let mut out = BufWriter::new(out);

    let svm_type = model.param.svm_type;

    // Print probability header / SVR info
    if predict_prob {
        if matches!(svm_type, SvmType::EpsilonSvr | SvmType::NuSvr) {
            if !quiet {
                let sigma = model.prob_a[0];
                eprintln!(
                    "Prob. model for test data: target value = predicted value + z,\n\
                     z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma={}",
                    format_g(sigma)
                );
            }
        } else if svm_type == SvmType::OneClass {
            writeln!(out, "labels normal outlier").unwrap();
        } else {
            // Classification header
            write!(out, "labels").unwrap();
            for &lab in &model.label {
                write!(out, " {}", lab).unwrap();
            }
            writeln!(out).unwrap();
        }
    }

    // Predict
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut error = 0.0;
    let (mut sump, mut sumt, mut sumpp, mut sumtt, mut sumpt) =
        (0.0, 0.0, 0.0, 0.0, 0.0);

    let use_prob_output =
        predict_prob && matches!(svm_type, SvmType::CSvc | SvmType::NuSvc | SvmType::OneClass);

    for (idx, instance) in problem.instances.iter().enumerate() {
        let target_label = problem.labels[idx];

        let predict_label = if use_prob_output {
            let (label, probs) = predict_probability(&model, instance)
                .expect("probability prediction failed");
            write!(out, "{}", format_g(label)).unwrap();
            for p in &probs {
                write!(out, " {}", format_g(*p)).unwrap();
            }
            writeln!(out).unwrap();
            label
        } else {
            let label = predict(&model, instance);
            writeln!(out, "{}", format_17g(label)).unwrap();
            label
        };

        if predict_label == target_label {
            correct += 1;
        }
        error += (predict_label - target_label) * (predict_label - target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label * predict_label;
        sumtt += target_label * target_label;
        sumpt += predict_label * target_label;
        total += 1;
    }

    if !quiet {
        if matches!(svm_type, SvmType::EpsilonSvr | SvmType::NuSvr) {
            let n = total as f64;
            eprintln!(
                "Mean squared error = {} (regression)",
                format_g(error / n)
            );
            let r2 = ((n * sumpt - sump * sumt) * (n * sumpt - sump * sumt))
                / ((n * sumpp - sump * sump) * (n * sumtt - sumt * sumt));
            eprintln!(
                "Squared correlation coefficient = {} (regression)",
                format_g(r2)
            );
        } else {
            eprintln!(
                "Accuracy = {}% ({}/{}) (classification)",
                format_g(correct as f64 / total as f64 * 100.0),
                correct,
                total
            );
        }
    }
}
