from dataclasses import dataclass, field
import yaml, os
import array

import numpy as np

def _(obj): 
    return field(default_factory=lambda: obj)

@dataclass
class test:

    lr: float = 0.001
    iters: int = 40
    hidden_sizes: list = _([32,32])

    def __post_init__(self):
        try:
            with open("./.test/example.yaml") as file:
                d = yaml.safe_load(file)

                for k, v in d.items():
                    print (str(k))
                    for key, value in v.items():
                        print(str(key) +":" + str(value))
                        v[key] = "Hallo"

                print(d['train_params'])

                with open ("Test.yaml", 'w') as file:
                    y = yaml.dump(d, file)

                self.__dict__.update(**d)
        except Exception as ex:
            print("using standard variables in datclass " + str(ex))


if __name__ == '__main__':
    my = test()
    print(my)

    # a = array.array('i',[1,2,3])
    # a.append(1)
    # print(type(a))

    # a = np.array(a)
    # print (type(a))

    # l = [1,2,3,4]

    # print(type(l)) # a = array.array('i',[1,2,3])
    # a.append(1)
    # print(type(a))

    # a = np.array(a)
    # print (type(a))

    # l = [1,2,3,4]

    # print(type(l))

    # l= np.array(l)

    # # l = l[None,:]
    # l = l.reshape(1,4)

    # print(type(l))

    # l = l.reshape(-1)

    # print(l)

    # no = np.array([0])
    # print(no)

    # l= np.array(l)

    # # l = l[None,:]
    # l = l.reshape(1,4)

    # print(type(l))

    # l = l.reshape(-1)

    # print(l)

    # no = np.array([0])
    # print(no)


    